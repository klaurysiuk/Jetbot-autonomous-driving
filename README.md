# JetBot Road Following

A deep-learning road-follower for the [NVIDIA JetBot](https://jetbot.org/). A
single forward camera frame is regressed into a 2-value target
`[distance, angle]` that describes "where on the road the robot should aim", and
a small steering controller turns that target into left/right motor commands.

The pipeline has three stages:

1. **Data collection** — drive the JetBot by hand and record its camera.
2. **Labeling & training** — annotate a target point on each frame and train a
   MobileNetV2 regressor.
3. **Deployment** — run the exported ONNX model live on the JetBot.

```
camera frame ──► MobileNetV2 ──► [distance, angle] ──► steering ──► L/R motors
```

---

## 1. Data collection

In the beginning, worth to be mentioned, that we didn't collect the data in a way it was proposed in NVIDIA notebooks, i.e. to collect X and Y values by joystick, thus we collected data by **manually driving the JetBot around the track** while
recording the onboard camera. There were two main reasons for this decision:

- Our first jetbot had a tremendous lag between command and action in around 10-20 seconds, so proper teleoperation wasn't possible

- We assumed, that even if we could teleoperate jetbot, we might have problems during data selection and cleaning, due to human mistakes during teleoperation 

Each driving session was saved as a separate
video file:

```
dataset/raw/videos/
  recording_20260428_145745.avi
  recording_20260428_145956.avi
  ...
```

The five `.avi` recordings (captured on 2026-04-28) are then split into
individual frames stored as PNG images:

```
dataset/raw/imgs/
  recording_20260428_145745_frame_000000.png
  recording_20260428_145745_frame_000001.png
  ...
```

This yields ~11.7k raw frames. Because consecutive frames are very similar, only
a subset needs to be labeled to get good coverage of the track (straights,
curves, intersections, different lighting). 

<img width="224" height="224" alt="image" src="https://github.com/user-attachments/assets/1220e3f4-9a6c-4d49-9f60-3e92b0a5c138" />


---

## 2. Labeling 

As it was mentioned earlier, our approach assumes that collected frames didn't have labels, thus we've used our own labeling script.
Each frame is labeled with a **single target point** — the spot on the road the robot should steer toward. Labeling is done with the interactive OpenCV tool
[manual_labeling.py](manual_labeling.py):

```bash
python manual_labeling.py
```
<img width="300" height="330" alt="image" src="https://github.com/user-attachments/assets/d78355d0-e32b-4717-a3d9-e6e2dfefea83" />


How it works:

- It scans `dataset/raw/imgs`, skips frames that are already labeled, shuffles
  the rest, and shows them one at a time.
- You **click** the target point on the road; a green cross marks it.
- Controls:

  | Key            | Action                                   |
  | -------------- | ---------------------------------------- |
  | Left click     | Set the target point                     |
  | `Enter`        | Confirm the point and go to next frame   |
  | `c`            | Clear the current point                  |
  | `Space`        | Skip the frame (don't label it)          |
  | `z`            | Undo the last label                      |
  | `p` / `Esc`    | Quit                                     |

- Confirming with no point stores `{"x": null, "y": null}` (frame has no valid
  target).
- Labels are written incrementally (a batch is saved every 10 confirmations,
  and again on exit) to:

```
dataset/raw/labeled/labels.json
```

Each entry maps an image file name to the clicked pixel coordinates, e.g.:

```json
{
  "recording_20260428_145745_frame_000123.png": { "x": 118, "y": 156 }
}
```

---

## 3. Training

Training lives in [mobilenet_traning.ipynb](mobilenet_traning.ipynb). It turns
the clicked pixel points into regression targets and fine-tunes a pretrained
**MobileNetV2**.

### 3.1 Targets: from pixel point to `[distance, angle]`

For each labeled frame (images are treated as `224 x 224`), the clicked point
`(x, y)` is converted into two normalized targets:

- **distance** — normalized forward progress, how "far up" the target sits:

  ```
  distance = 1 - (y / 224)
  ```

- **angle** — steering angle toward the point, measured from straight-ahead:

  ```
  angle = atan2(x - W/2, H - y)        # W = H = 224
  ```

  All angles are then normalized by the largest absolute angle and sign-flipped,
  so they land in roughly `[-1, 1]`.

Frames with `x/y = null` are dropped. The result is written to
`dataset/raw/labeled/labels.csv` (`file_name, distance, angle`) — **972 labeled
samples** in our run.

### 3.2 Dataset & preprocessing

`XYDataset` loads each image and applies the same preprocessing the deployment
pipeline expects:

- `ColorJitter(0.3, 0.3, 0.3, 0.3)` for color robustness,
- resize to `224 x 224`,
- `to_tensor`,
- RGB → BGR channel flip,
- ImageNet normalization (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).

**Horizontal-flip augmentation** is enabled: with 50% probability the image is
mirrored and the `angle` sign is flipped, doubling the effective left/right
coverage of the track.

### 3.3 Model

- Backbone: `torchvision` **MobileNetV2** with ImageNet-pretrained weights.
- The classifier head is replaced with a single linear layer producing **2
  outputs** (`distance`, `angle`):

  ```python
  model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
  model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
  ```

  ~2.2M trainable parameters.

### 3.4 Training loop

- Split: 90% train / 10% test (`875 / 97` samples), batch size 16.
- Optimizer: Adam, `lr = 0.001`.
- Loss: MSE over both outputs.
- 15 epochs; the checkpoint with the **lowest test MSE** is kept.

Results for the final model: **overall test MSE ≈ 0.037** (distance ≈ 0.019, angle
≈ 0.055).

### 3.5 Export

The best checkpoint is saved as both:

- `jetbot_cnn_mobilenet_v2.pth` — PyTorch state-dict (used by
  [show_results.py](show_results.py) for offline evaluation), and
- `jetbot_cnn_mobilenet_v2.onnx` — exported with `opset 11`, fixed
  `1x3x224x224` input plus a dynamic batch axis, for fast inference on the
  JetBot via ONNX Runtime / TensorRT.

Trained models live under [models/](models/).

### 3.6 Offline check

Before deploying, predictions can be previewed on the recorded videos:

```bash
python show_results.py                 # first video in dataset/raw/videos
python show_results.py --video recording_20260428_150218.avi
python show_results.py --list
```

It overlays the predicted `distance`/`angle` and draws a heading arrow on each
frame (Space = pause, Right/`d` = step, `Q`/`Esc` = quit). It auto-detects the
architecture (ResNet-18 / MobileNetV2 / SmallCNN) from the checkpoint keys.

<img width="426" height="240" alt="Video Project 3" src="https://github.com/user-attachments/assets/92412d9e-1dd6-46c3-a72e-ab83885c50e4" />

---

## 4. Running on the JetBot

Live road following runs on the robot via
[road_following_onnx.py](road_following_onnx.py):

```bash
python road_following_onnx.py
```

What it does each camera frame:

1. **Preprocess** — convert the frame to a tensor and apply ImageNet
   normalization (using FP16 on CUDA for speed).
2. **Infer** — run the ONNX model (`jetbot_cnn_mobilenet_v2.onnx`) through ONNX
   Runtime, preferring `CUDAExecutionProvider` and falling back to CPU. The
   output is `[distance, angle]`.
3. **Compute heading** — interpret the outputs as a target direction:

   ```python
   x = -xy[1]   # angle component
   y =  xy[0]   # distance component
   angle = atan2(x, y)
   ```

4. **Steering controller** — a P(+D) controller converts the angle into a
   steering signal:

   ```
   steering = angle * STEERING_GAIN + (angle - angle_last) * STEERING_DGAIN + STEERING_BIAS
   ```

5. **Differential drive** — the steering is mixed into left/right wheel speeds
   around a constant base speed, then normalized/clipped to `[-1, 1]`:

   ```python
   left  = BASE_SPEED + steering
   right = BASE_SPEED - steering
   ```

The controller is driven by the JetBot `Camera` observer, so it reacts to every
new frame. `Ctrl+C` (SIGINT/SIGTERM) cleanly stops the camera and motors.

In our case, processing frequency amounted to ~7-8 Hz for .onnx model and ~3-4 Hz for .pth model

#### Tuning

The constants at the top of [road_following_onnx.py](road_following_onnx.py)
control the behavior:

| Constant         | Values   | Meaning                                        |
| ---------------- | -------- | ---------------------------------------------- |
| `BASE_SPEED`     | `0.3`    | Forward speed of both wheels                   |
| `STEERING_GAIN`  | `0.11`   | Proportional steering strength                 |
| `STEERING_DGAIN` | `0.08`   | Derivative (damping) term                      |
| `STEERING_BIAS`  | `0.0`    | Constant steering offset (mechanical trim)     |
| `MODEL_PATH`     | —        | Path to the `.onnx` model to load              |

With these values Jetbot passed a big track by 12 seconds and a small one by 8 seconds 

---

## 5. Other approaches, Error analysis, and conclusions 

### Alternative approach  
During the project, we had a couple of ideas we tried to implement. Recalling the format of our raw data, the first idea we implemented was finding **X** and **Y** by **tracking the road surface markings** — more exactly, the angle between the robot's camera and the lines, and the distance to the end of the road or to the next turn. First, we implemented a classical CV pipeline with **Hough lines**; later, we wanted to use a lightweight segmentation model (particularly **YOLO11-n**), trained on marking masks produced by **SAM 2**, to segment markings online and process masks rather than raw frames from the camera. However, this approach wasn't as successful as the currently implemented one.

### Model and PID controller 

Speaking about the model, in the beginning, in the beginning we chose **ResNet-18** as a backbone. However, it was too heavy and slow (processing frequency was ~0.3 Hz), so we changed it to the final **MobileNetV2**. As written earlier, **ONNX**  compression also helped in increasing the processing frequency.

Finally, finding proper values for **PID controller** wasn't big problem, we've done it empirically during a couple of tests on track.

### Hardware
Particular mention should be made of problems with hardware. First of all, our first two jetbots had problems with drivers, so we had to reflash it. Second, in the middle of tests, our jetbot's wheel axle was damaged, so we had to change it. Additionally, camera calibration should have been done each time we started autonomous driving.

### Reproducibility 

Our script is  designed to run on any JetBot setup, supporting both GPU-accelerated and CPU-only inference. It maintains a low memory footprint to prevent out-of-memory crashes and requires minimal storage space. We successfully tested our code directly on the native OS across multiple JetBots with varying hardware configurations.

When deploying on devices without a GPU, the inference rate naturally decreases; to maintain stable road following, we adjusted the PID controller parameters (e.g., reducing the base speed). Despite these hardware differences, the core neural network model remained completely unchanged.

Camera setup was straightforward; however, we strictly maintained the camera's tilt angle relative to the ground to match the one used during dataset creation (~35-40 degrees). Additionally, we tested our script across various lighting conditions, confirming that the model performs robustly in both bright and dim environments.

Both the pre-trained weights and the complete dataset are available for download. Full details regarding the model's training process, hyperparameters, and reproducibility can be found in the provided training notebook ([mobilenet_training.ipynb](./mobilenet_traning.ipynb)).

### Conclusions

We created pipeline for autonomous jetbot operation. We've collected our own dataset, implemented different strategies, and became a recordists among other teams. Our best time of 11.98 seconds on a big track shows that we created competitive workflow.

<img width="240" height="240" alt="0618" src="https://github.com/user-attachments/assets/9c996989-4b02-4055-912c-2c739fc297c8" />

## Repository layout

```
.
├── manual_labeling.py        # interactive point-labeling tool (raw frames -> labels.json)
├── mobilenet_traning.ipynb    # build labels.csv, train MobileNetV2, export .pth/.onnx
├── show_results.py            # offline: overlay predictions on the recorded videos
├── road_following_onnx.py     # live road following on the JetBot (ONNX inference)
├── models/                    # trained checkpoints (.pth) and exported .onnx models
└── dataset/
    ├── raw/
    │   ├── videos/            # manually-driven camera recordings (.avi)
    │   ├── imgs/              # frames extracted from the recordings (.png)
    │   └── labeled/           # labels.json (clicked points) + labels.csv (training targets)
    └── put_jetbot_dataset/    # external PUT JetBot reference dataset
```

## Requirements

- **Training / offline tools (PC):** Python 3, PyTorch, torchvision, OpenCV,
  Pillow, NumPy, pandas, onnx, onnxruntime, tqdm.
- **JetBot (deployment):** the `jetbot` package, PyTorch + torchvision,
  onnxruntime (GPU build recommended), Pillow, NumPy.

