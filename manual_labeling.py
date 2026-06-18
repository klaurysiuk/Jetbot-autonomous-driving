import json
import random
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).resolve().parent
IMGS_DIR = PROJECT_DIR / "dataset" / "raw" / "imgs"
LABELED_DIR = PROJECT_DIR / "dataset" / "raw" / "labeled"
LABELS_FILE = LABELED_DIR / "labels.json"

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
QUIT_KEYS = {ord("p"), 27}  # 'p' or Esc
SKIP_KEY = ord(" ")
UNDO_KEY = ord("z")
CONFIRM_KEYS = {13, 10}  # Enter
CLEAR_KEY = ord("c")
SAVE_BATCH_SIZE = 10
WINDOW = "manual labeling - click point, [enter] confirm, [c] clear, [space] skip, [z] undo, [p] quit"


def load_labels() -> dict:
    if LABELS_FILE.exists():
        with LABELS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_labels(labels: dict) -> None:
    LABELED_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LABELS_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    tmp.replace(LABELS_FILE)


def draw_hud(img, idx: int, total: int, name: str, last_label: str | None,
             point: tuple[int, int] | None, already_labeled: int = 0,
             total_all: int | None = None):
    img = img.copy()
    if point is not None:
        x, y = point
        cv2.drawMarker(img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(img, (x, y), 6, (0, 255, 0), 1)
        cv2.putText(img, f"({x}, {y})", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    overlay = img.copy()
    h = img.shape[0]
    cv2.rectangle(overlay, (0, 0), (img.shape[1], 60), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, h - 30), (img.shape[1], h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    overall_idx = already_labeled + idx + 1
    overall_total = total_all if total_all is not None else already_labeled + total
    counter = f"{overall_idx}/{overall_total} (session {idx + 1}/{total})"
    cv2.putText(img, f"{counter}  {name}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"last: {last_label or '-'}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "click=set point  enter=confirm  c=clear  space=skip  z=undo  p=quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    if not IMGS_DIR.is_dir():
        raise SystemExit(f"Images dir not found: {IMGS_DIR}")

    images = [p for p in IMGS_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not images:
        raise SystemExit(f"No images found in {IMGS_DIR}")

    labels = load_labels()
    total_found = len(images)
    images = [p for p in images if p.name not in labels]
    already_labeled = total_found - len(images)
    print(f"Found {total_found} images: {already_labeled} already labeled, "
          f"{len(images)} remaining.")
    if not images:
        raise SystemExit("Nothing left to label.")
    random.shuffle(images)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    state = {"point": None}

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["point"] = (x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)

    history: list[str] = []
    last_label: str | None = None
    quit_requested = False
    pending_since_save = 0

    try:
        for idx, path in enumerate(images):
            img = cv2.imread(str(path))
            if img is None:
                print(f"Failed to read {path.name}, skipping.")
                continue

            state["point"] = None

            while True:
                cv2.imshow(WINDOW, draw_hud(img, idx, len(images), path.name,
                                            last_label, state["point"],
                                            already_labeled, total_found))
                key = cv2.waitKey(20) & 0xFF

                if key == 255:
                    # no key pressed; refresh to show new mouse clicks
                    continue
                if key in QUIT_KEYS:
                    quit_requested = True
                    break
                if key in CONFIRM_KEYS:
                    if state["point"] is None:
                        labels[path.name] = {"x": None, "y": None}
                        last_label = "(none)"
                    else:
                        x, y = state["point"]
                        labels[path.name] = {"x": int(x), "y": int(y)}
                        last_label = f"({x}, {y})"
                    history.append(path.name)
                    pending_since_save += 1
                    if pending_since_save >= SAVE_BATCH_SIZE:
                        save_labels(labels)
                        print(f"Saved batch of {pending_since_save} ({len(labels)} total).")
                        pending_since_save = 0
                    break
                if key == CLEAR_KEY:
                    state["point"] = None
                    continue
                if key == SKIP_KEY:
                    last_label = "skipped"
                    break
                if key == UNDO_KEY:
                    if history:
                        prev = history.pop()
                        labels.pop(prev, None)
                        pending_since_save = max(0, pending_since_save - 1)
                        last_label = f"undone {prev}"
                        print(f"Undone label for {prev}.")
                    else:
                        last_label = "nothing to undo"
                    continue
                # any other key: re-prompt
            if quit_requested:
                break
    finally:
        cv2.destroyAllWindows()
        save_labels(labels)
        print(f"Saved {len(labels)} labels to {LABELS_FILE}.")


if __name__ == "__main__":
    main()
