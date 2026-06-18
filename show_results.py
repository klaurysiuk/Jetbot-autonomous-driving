"""Show road-follower predictions on a video, frame by frame.

Usage:
    python show_results.py                 # picks the first video automatically
    python show_results.py --video <name>  # use a specific .avi from dataset/raw/videos
    python show_results.py --list          # list available videos

Controls while playing:
    Space  - pause / resume
    Right  - step one frame (when paused)
    Q/Esc  - quit
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "dataset" / "raw" / "videos"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "jetbot_cnn_mobilenet_v2.pth"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SmallCNN(torch.nn.Module):
    """Mirror of the architecture in train_model2.ipynb (500k params)."""

    def __init__(self, num_outputs=2):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
            )

        self.features = torch.nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
            conv_block(128, 224),
        )
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(224, num_outputs)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _build_resnet18(num_outputs: int = 2) -> torch.nn.Module:
    model = tv_models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, num_outputs)
    return model


def _build_mobilenet_v2(num_outputs: int = 2) -> torch.nn.Module:
    model = tv_models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_outputs)
    return model


def load_model(model_path: Path, device: torch.device) -> torch.nn.Module:
    state = torch.load(str(model_path), map_location=device)
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    # Auto-detect architecture from state-dict keys.
    keys = set(state.keys())
    if {"conv1.weight", "layer1.0.conv1.weight", "fc.weight"}.issubset(keys):
        model = _build_resnet18(num_outputs=2)
        arch_name = "ResNet-18"
    elif {"features.0.0.weight", "classifier.1.weight"}.issubset(keys):
        model = _build_mobilenet_v2(num_outputs=2)
        arch_name = "MobileNetV2"
    else:
        model = SmallCNN(num_outputs=2)
        arch_name = "SmallCNN (500k)"

    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Arch:   {arch_name}")
    return model


def preprocess_frame(frame_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Match the XYDataset preprocessing: PIL -> resize -> to_tensor -> BGR -> normalize."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(rgb)
    image = transforms.functional.resize(image, (IMG_SIZE, IMG_SIZE))
    image = transforms.functional.to_tensor(image)
    image = image.numpy()[::-1].copy()  # RGB -> BGR (training did the same)
    image = torch.from_numpy(image)
    image = transforms.functional.normalize(image, MEAN, STD)
    return image.unsqueeze(0).to(device)


def draw_overlay(frame: np.ndarray, distance: float, angle: float, title: str) -> np.ndarray:
    """Render the prediction onto a copy of the frame."""
    h, w = frame.shape[:2]
    canvas = frame.copy()

    panel_h = 110
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), thickness=-1)
    canvas = cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)

    angle_deg = math.degrees(float(angle))
    cv2.putText(canvas, title,
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"distance: {float(distance):+.3f}",
                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"angle:    {float(angle):+.3f}  ({angle_deg:+6.1f} deg)",
                (12, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)

    # Arrow representing predicted heading: starts at bottom-center, length scales with distance.
    cx = w // 2
    cy = h - 20
    length = int(0.45 * h * max(0.05, min(1.0, float(distance))))
    end_x = int(cx + length * math.sin(float(angle)))
    end_y = int(cy - length * math.cos(float(angle)))
    cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), (0, 255, 0), 4, tipLength=0.25)
    cv2.circle(canvas, (cx, cy), 6, (0, 255, 0), -1)
    return canvas


def list_videos(videos_dir: Path) -> list[Path]:
    if not videos_dir.exists():
        return []
    return sorted(p for p in videos_dir.iterdir()
                  if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"})


def resolve_video(arg_value: str | None, videos_dir: Path) -> Path:
    videos = list_videos(videos_dir)
    if not videos:
        sys.exit(f"No videos found in {videos_dir}")
    if arg_value is None:
        return videos[0]
    candidate = Path(arg_value)
    if candidate.is_file():
        return candidate
    candidate = videos_dir / arg_value
    if candidate.is_file():
        return candidate
    sys.exit(f"Video not found: {arg_value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=str, default=None,
                        help="Video file name (inside dataset/raw/videos) or absolute path.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL_PATH),
                        help="Path to the .pth weights file.")
    parser.add_argument("--videos-dir", type=str, default=str(DEFAULT_VIDEOS_DIR),
                        help="Directory containing the videos.")
    parser.add_argument("--list", action="store_true", help="List available videos and exit.")
    parser.add_argument("--display-width", type=int, default=720,
                        help="Width to scale the display window to (keeps aspect ratio).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override playback FPS (defaults to the video's native FPS).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)

    if args.list:
        videos = list_videos(videos_dir)
        if not videos:
            print(f"No videos in {videos_dir}")
            return
        print(f"Videos in {videos_dir}:")
        for v in videos:
            print(f"  - {v.name}")
        return

    video_path = resolve_video(args.video, videos_dir)
    model_path = Path(args.model)
    if not model_path.is_file():
        sys.exit(f"Model file not found: {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model:  {model_path}")
    print(f"Video:  {video_path}")

    model = load_model(model_path, device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"Failed to open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = args.fps if args.fps else native_fps
    frame_delay_ms = max(1, int(round(1000.0 / target_fps)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_name = f"JetBot CNN - {video_path.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    paused = False
    frame_idx = 0
    last_pred = (0.0, 0.0)

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    print("End of video.")
                    break
                frame_idx += 1

                with torch.no_grad():
                    inp = preprocess_frame(frame, device)
                    out = model(inp).squeeze(0).cpu().numpy()
                last_pred = (float(out[0]), float(out[1]))

            distance, angle = last_pred
            angle = -angle
            shown = draw_overlay(frame, distance, angle, title=f"Model: {model_path.name}")

            if args.display_width and shown.shape[1] != args.display_width:
                scale = args.display_width / shown.shape[1]
                new_size = (args.display_width, int(shown.shape[0] * scale))
                shown = cv2.resize(shown, new_size, interpolation=cv2.INTER_LINEAR)

            status = "PAUSED" if paused else "PLAYING"
            cv2.putText(shown, f"{status}  frame {frame_idx}/{total_frames}",
                        (12, shown.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, shown)

            key = cv2.waitKey(frame_delay_ms if not paused else 0) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            if key == ord(" "):
                paused = not paused
            elif paused and key in (83, ord("d"), ord("n")):  # Right arrow / d / n -> single step
                ok, frame = cap.read()
                if not ok:
                    print("End of video.")
                    break
                frame_idx += 1
                with torch.no_grad():
                    inp = preprocess_frame(frame, device)
                    out = model(inp).squeeze(0).cpu().numpy()
                last_pred = (float(out[0]), float(out[1]))
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
