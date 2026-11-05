from pathlib import Path
import threading
import time

import cv2
import torch
from ultralytics import YOLO


# =========================
# CONFIG - EDIT THESE
# =========================

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = ROOT_DIR / "videos" / "30fps.mp4"

MODEL_FILE = "yolo26s.pt"
IMGSZ = 1920


# =========================
# DEPLOYMENT SETTINGS
# =========================

DEVICE = 0
CONFIDENCE = 0.25

# person, car, airplane
COCO_CLASSES = [0, 2, 4]

RECT = True
QUANTIZE = 32

WARMUP_RUNS = 3

DISPLAY_SCALE = 1.0
WINDOW_NAME = "YOLO26 Viewer"


class LatestFrameBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = -1
        self.produced = 0
        self.finished = False

    def publish(self, frame_id, frame):
        with self.condition:
            self.frame_id = frame_id
            self.frame = frame
            self.produced += 1
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()

    def get_latest(self, last_frame_id):
        with self.condition:
            self.condition.wait_for(
                lambda:
                    self.frame_id > last_frame_id
                    or self.finished
            )

            if self.frame_id > last_frame_id:
                return self.frame_id, self.frame

            return None


def video_producer(
    video_path,
    source_fps,
    buffer,
    stop_event,
):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        buffer.finish()
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    period = 1.0 / source_fps
    start_time = time.perf_counter()
    frame_id = 0

    while not stop_event.is_set():
        ret, frame = cap.read()

        if not ret:
            break

        target_time = (
            start_time
            + frame_id * period
        )

        delay = (
            target_time
            - time.perf_counter()
        )

        if delay > 0:
            time.sleep(delay)

        if stop_event.is_set():
            break

        buffer.publish(
            frame_id,
            frame,
        )

        frame_id += 1

    cap.release()
    buffer.finish()


def warmup(
    model,
    frame,
):
    print(
        f"Warming up {WARMUP_RUNS} runs..."
    )

    for _ in range(
        WARMUP_RUNS
    ):
        model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONFIDENCE,
            classes=COCO_CLASSES,
            device=DEVICE,
            rect=RECT,
            quantize=QUANTIZE,
            verbose=False,
        )

    torch.cuda.synchronize()


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            VIDEO_PATH
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available."
        )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    source_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    ret, warmup_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(
            "Could not read a warmup frame."
        )

    if source_fps <= 0:
        raise RuntimeError(
            "Invalid source FPS."
        )

    print("=" * 70)
    print("YOLO26 STANDALONE VIEWER")
    print("=" * 70)
    print(f"Video:       {VIDEO_PATH}")
    print(
        f"Source:      "
        f"{source_width}x{source_height} "
        f"@ {source_fps:.3f} FPS"
    )
    print(f"Model:       {MODEL_FILE}")
    print(f"Resolution:  {IMGSZ}")
    print(f"Confidence:  {CONFIDENCE}")
    print(f"Classes:     {COCO_CLASSES}")
    print("Controls:    q or Esc = quit")

    model = YOLO(
        MODEL_FILE
    )

    warmup(
        model,
        warmup_frame,
    )

    buffer = LatestFrameBuffer()
    stop_event = threading.Event()

    producer_thread = threading.Thread(
        target=video_producer,
        args=(
            VIDEO_PATH,
            source_fps,
            buffer,
            stop_event,
        ),
        daemon=True,
    )

    producer_thread.start()

    last_frame_id = -1
    processed = 0
    viewer_start = time.perf_counter()

    while True:
        item = buffer.get_latest(
            last_frame_id
        )

        if item is None:
            break

        frame_id, frame = item
        last_frame_id = frame_id

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONFIDENCE,
            classes=COCO_CLASSES,
            device=DEVICE,
            rect=RECT,
            quantize=QUANTIZE,
            verbose=False,
        )[0]

        torch.cuda.synchronize()

        predict_ms = (
            time.perf_counter()
            - start
        ) * 1000.0

        processed += 1

        elapsed = max(
            time.perf_counter()
            - viewer_start,
            1e-9,
        )

        processed_fps = (
            processed / elapsed
        )

        annotated = result.plot()

        status = (
            f"{Path(MODEL_FILE).stem} | "
            f"{IMGSZ} | "
            f"{predict_ms:.1f} ms | "
            f"{processed_fps:.1f} processed FPS"
        )

        cv2.putText(
            annotated,
            status,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if DISPLAY_SCALE != 1.0:
            annotated = cv2.resize(
                annotated,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_AREA,
            )

        cv2.imshow(
            WINDOW_NAME,
            annotated,
        )

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key in (
            ord("q"),
            27,
        ):
            stop_event.set()
            break

    stop_event.set()
    producer_thread.join()
    cv2.destroyAllWindows()

    print()
    print(
        f"Frames produced:  "
        f"{buffer.produced}"
    )
    print(
        f"Frames displayed: "
        f"{processed}"
    )
    print(
        f"Frames skipped:   "
        f"{max(buffer.produced - processed, 0)}"
    )


if __name__ == "__main__":
    main()
