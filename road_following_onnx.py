#!/usr/bin/env python3

import os
import signal
import sys
import time

import numpy as np
import PIL.Image
import torch
import torchvision
import torchvision.transforms as transforms

from jetbot import Camera, Robot


MODEL_PATH = "jetbot_cnn_mobilenet_v2.pth"

# ----------------------------
# CONFIG
# ----------------------------
BASE_SPEED = 0.3
STEERING_GAIN = 0.11
STEERING_DGAIN = 0.08
STEERING_BIAS = 0.0

LOG_EVERY = 1.0


# ----------------------------
# MODEL
# ----------------------------
def load_model(model_path: str, device: torch.device):
    model = torchvision.models.mobilenet_v2(pretrained=False)  # FIX for JetBot
    model.classifier[1] = torch.nn.Linear(1280, 2)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model = model.to(device)
    model = model.eval()

    # FP16 optional (safer to disable if unstable)
    if device.type == "cuda":
        model = model.half()

    return model


# ----------------------------
# MAIN CLASS
# ----------------------------
class RoadFollower:

    def __init__(
        self,
        model_path,
        base_speed,
        steering_gain,
        steering_dgain,
        steering_bias,
        log_every=1.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[init] device = {self.device}")

        self.model = load_model(model_path, self.device)

        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        if self.device.type == "cuda":
            self.mean = self.mean.half()
            self.std = self.std.half()

        self.base_speed = base_speed
        self.steering_gain = steering_gain
        self.steering_dgain = steering_dgain
        self.steering_bias = steering_bias

        self.angle = 0.0
        self.angle_last = 0.0

        self._log_every = log_every
        self._last_log = 0.0

        print("[init] starting camera...")
        self.camera = Camera()

        print("[init] starting robot...")
        self.robot = Robot()

    # ----------------------------
    # IMAGE PREPROCESS
    # ----------------------------
    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(self.device)

        if self.device.type == "cuda":
            image = image.half()

        image = (image - self.mean[:, None, None]) / self.std[:, None, None]

        return image.unsqueeze(0)

    # ----------------------------
    # MAIN CALLBACK
    # ----------------------------
    def execute(self, change):
        image = change["new"]

        xy = self.model(self.preprocess(image)).detach().float().cpu().numpy().flatten()

        x = -float(xy[1]) # angle
        y = float(xy[0]) # distance

        # ----------------------------
        # Steering (PID-like P + D)
        # ----------------------------
        self.angle = float(np.arctan2(x, y))

        steering = (
            self.angle * self.steering_gain
            + (self.angle - self.angle_last) * self.steering_dgain
        )

        self.angle_last = self.angle

        steering += self.steering_bias

        # ----------------------------
        # Differential drive (FIXED)
        # ----------------------------
        left = self.base_speed + steering
        right = self.base_speed - steering

        # Normalize instead of killing one wheel
        max_mag = max(abs(left), abs(right), 1.0)
        left /= max_mag
        right /= max_mag

        left = float(np.clip(left, -1.0, 1.0))
        right = float(np.clip(right, -1.0, 1.0))

        self.robot.left_motor.value = left
        self.robot.right_motor.value = right

        # ----------------------------
        # LOGGING
        # ----------------------------
        now = time.time()
        if now - self._last_log > self._log_every:
            print(
                f"[telemetry] angle={self.angle:+.3f} "
                f"steering={steering:+.3f} "
                f"x(angle)={x} y(distance)={y}",
                f"L={left:+.3f} R={right:+.3f}"
            )
            self._last_log = now

    # ----------------------------
    # START / STOP
    # ----------------------------
    def start(self):
        self.execute({"new": self.camera.value})
        self.camera.observe(self.execute, names="value")
        print("[run] active — Ctrl+C to stop")

    def stop(self):
        print("[stop] shutting down...")

        try:
            self.camera.unobserve(self.execute, names="value")
        except Exception:
            pass

        time.sleep(0.1)

        try:
            self.robot.stop()
        except Exception:
            pass

        try:
            self.camera.stop()
        except Exception:
            pass

        print("[stop] done")


# ----------------------------
# MAIN
# ----------------------------
def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"[fatal] model not found: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"[config] model = {MODEL_PATH}")
    print(f"[config] base_speed = {BASE_SPEED}")
    print(f"[config] steering_gain = {STEERING_GAIN}")

    follower = RoadFollower(
        MODEL_PATH,
        BASE_SPEED,
        STEERING_GAIN,
        STEERING_DGAIN,
        STEERING_BIAS,
        log_every=LOG_EVERY,
    )

    def shutdown(sig, _frame):
        print(f"\n[signal] {sig}")
        follower.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    follower.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
