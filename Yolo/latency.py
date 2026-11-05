import csv
import gc
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

from experiment_config import (
    MODELS,
    RESOLUTIONS,
    DEVICE,
    QUANTIZE,
    RECT,
    LATENCY_CONF,
    WARMUP_RUNS,
    SHOW_VIEWER,
    WINDOW_NAME,
    VIDEO_PATH,
    LATENCY_OUTPUT_DIR,
    LATENCY_SUMMARY_FILE,
    COCO_EVAL_CLASSES,
)


class LatestFrameBuffer:

    def __init__(
        self,
    ):

        self.condition = (
            threading.Condition()
        )

        self.frame = None
        self.frame_id = -1
        self.arrival_time = None

        self.produced = 0
        self.finished = False
        self.playback_start = None

    def publish(
        self,
        frame,
        frame_id,
    ):

        with self.condition:

            self.frame = frame
            self.frame_id = frame_id
            self.arrival_time = (
                time.perf_counter()
            )

            self.produced += 1

            self.condition.notify_all()

    def finish(
        self,
    ):

        with self.condition:

            self.finished = True

            self.condition.notify_all()

    def get_latest(
        self,
        last_frame_id,
    ):

        with self.condition:

            self.condition.wait_for(
                lambda:
                    self.frame_id
                    > last_frame_id
                    or self.finished
            )

            if (
                self.frame_id
                > last_frame_id
            ):

                return (
                    self.frame_id,
                    self.frame,
                    self.arrival_time,
                )

            return None


def video_producer(
    video_path,
    frame_buffer,
    stop_event,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        frame_buffer.finish()

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if source_fps <= 0:

        cap.release()
        frame_buffer.finish()

        raise RuntimeError(
            "Invalid source video FPS."
        )

    frame_period = (
        1.0
        / source_fps
    )

    playback_start = (
        time.perf_counter()
    )

    frame_buffer.playback_start = (
        playback_start
    )

    frame_id = 0

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            break

        target_time = (
            playback_start
            + frame_id
            * frame_period
        )

        delay = (
            target_time
            - time.perf_counter()
        )

        if delay > 0:
            time.sleep(
                delay
            )

        if stop_event.is_set():
            break

        frame_buffer.publish(
            frame,
            frame_id,
        )

        frame_id += 1

    cap.release()

    frame_buffer.finish()


def warmup_model(
    model,
    warmup_frame,
    imgsz,
):

    print(
        f"  Warming up with "
        f"{WARMUP_RUNS} frames..."
    )

    for _ in range(
        WARMUP_RUNS
    ):

        model.predict(
            source=warmup_frame,
            imgsz=imgsz,
            device=DEVICE,
            quantize=QUANTIZE,
            rect=RECT,
            conf=LATENCY_CONF,
            classes=COCO_EVAL_CLASSES,
            verbose=False,
        )

    torch.cuda.synchronize()

    print(
        "  Warmup complete."
    )


def run_test(
    model_name,
    model,
    imgsz,
    warmup_frame,
):

    print()
    print("=" * 70)

    print(
        f"{model_name.upper()} "
        f"@ {imgsz}"
    )

    print("=" * 70)

    warmup_model(
        model,
        warmup_frame,
        imgsz,
    )

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    source_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
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

    cap.release()

    if source_fps <= 0:
        raise RuntimeError(
            "Invalid source FPS."
        )

    frame_period_ms = (
        1000.0
        / source_fps
    )

    print(
        f"  Source resolution: "
        f"{source_width}x"
        f"{source_height}"
    )

    print(
        f"  Source FPS:        "
        f"{source_fps:.3f}"
    )

    print(
        f"  Frame period:      "
        f"{frame_period_ms:.3f} ms"
    )

    frame_buffer = (
        LatestFrameBuffer()
    )

    stop_event = (
        threading.Event()
    )

    producer_thread = (
        threading.Thread(
            target=video_producer,
            args=(
                VIDEO_PATH,
                frame_buffer,
                stop_event,
            ),
            daemon=True,
        )
    )

    producer_thread.start()

    while (
        frame_buffer.playback_start
        is None
        and producer_thread.is_alive()
    ):
        time.sleep(
            0.001
        )

    if (
        frame_buffer.playback_start
        is None
    ):
        raise RuntimeError(
            "Video producer failed "
            "to start."
        )

    run_start = (
        frame_buffer.playback_start
    )

    predict_times = []
    ultralytics_inference_times = []
    e2e_times = []

    processed = 0
    last_frame_id = -1

    while True:

        item = (
            frame_buffer.get_latest(
                last_frame_id
            )
        )

        if item is None:
            break

        (
            frame_id,
            frame_bgr,
            arrival_time,
        ) = item

        last_frame_id = (
            frame_id
        )

        # Complete YOLO predict path:
        # preprocessing -> H2D -> network -> postprocessing.
        torch.cuda.synchronize()

        predict_start = (
            time.perf_counter()
        )

        result = model.predict(
            source=frame_bgr,
            imgsz=imgsz,
            device=DEVICE,
            quantize=QUANTIZE,
            rect=RECT,
            conf=LATENCY_CONF,
            classes=COCO_EVAL_CLASSES,
            verbose=False,
        )[0]

        torch.cuda.synchronize()

        predict_end = (
            time.perf_counter()
        )

        predict_ms = (
            predict_end
            - predict_start
        ) * 1000.0

        predict_times.append(
            predict_ms
        )

        # Keep Ultralytics' internal inference-only value as an
        # additional diagnostic, but do not use it as the primary
        # cross-family predict latency.
        ultralytics_inference_times.append(
            float(
                result.speed[
                    "inference"
                ]
            )
        )

        key = -1

        if SHOW_VIEWER:

            annotated = (
                result.plot()
            )

            cv2.imshow(
                (
                    f"{WINDOW_NAME} - "
                    f"{model_name} "
                    f"@ {imgsz}"
                ),
                annotated,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

        completion_time = (
            time.perf_counter()
        )

        e2e_ms = (
            completion_time
            - arrival_time
        ) * 1000.0

        e2e_times.append(
            e2e_ms
        )

        processed += 1

        if (
            processed % 500
            == 0
        ):

            print(
                f"  Processed "
                f"{processed} | "
                f"Predict "
                f"{predict_ms:.2f} ms | "
                f"E2E "
                f"{e2e_ms:.2f} ms"
            )

        if key == ord(
            "q"
        ):

            stop_event.set()

            break

    stop_event.set()

    producer_thread.join()

    run_end = (
        time.perf_counter()
    )

    if SHOW_VIEWER:
        cv2.destroyAllWindows()

    produced = (
        frame_buffer.produced
    )

    dropped = max(
        produced
        - processed,
        0,
    )

    if processed == 0:
        raise RuntimeError(
            "No frames were processed."
        )

    predict_times = np.asarray(
        predict_times,
        dtype=np.float64,
    )

    ultralytics_inference_times = np.asarray(
        ultralytics_inference_times,
        dtype=np.float64,
    )

    e2e_times = np.asarray(
        e2e_times,
        dtype=np.float64,
    )

    elapsed = (
        run_end
        - run_start
    )

    effective_fps = (
        processed
        / elapsed
    )

    drop_percent = (
        100.0
        * dropped
        / produced
        if produced > 0
        else 0.0
    )

    result_row = {
        "model":
            model_name,

        "imgsz":
            imgsz,

        "source_width":
            source_width,

        "source_height":
            source_height,

        "source_fps":
            source_fps,

        "source_frames":
            source_frames,

        "frame_period_ms":
            frame_period_ms,

        "frames_produced":
            produced,

        "frames_processed":
            processed,

        "frames_dropped":
            dropped,

        "drop_percent":
            drop_percent,

        "effective_fps":
            effective_fps,

        "predict_mean_ms":
            float(
                np.mean(
                    predict_times
                )
            ),

        "predict_median_ms":
            float(
                np.median(
                    predict_times
                )
            ),

        "predict_p95_ms":
            float(
                np.percentile(
                    predict_times,
                    95,
                )
            ),

        "predict_p99_ms":
            float(
                np.percentile(
                    predict_times,
                    99,
                )
            ),

        "e2e_mean_ms":
            float(
                np.mean(
                    e2e_times
                )
            ),

        "e2e_median_ms":
            float(
                np.median(
                    e2e_times
                )
            ),

        "e2e_p95_ms":
            float(
                np.percentile(
                    e2e_times,
                    95,
                )
            ),

        "e2e_p99_ms":
            float(
                np.percentile(
                    e2e_times,
                    99,
                )
            ),

        "e2e_max_ms":
            float(
                np.max(
                    e2e_times
                )
            ),

        "deadline_miss_percent":
            float(
                100.0
                * np.mean(
                    e2e_times
                    > frame_period_ms
                )
            ),

        "ultralytics_inference_mean_ms":
            float(
                np.mean(
                    ultralytics_inference_times
                )
            ),
    }

    print()
    print("  Results")
    print("  -------")

    print(
        f"  Produced:             "
        f"{produced}"
    )

    print(
        f"  Processed:            "
        f"{processed}"
    )

    print(
        f"  Dropped:              "
        f"{dropped}"
    )

    print(
        f"  Drop rate:            "
        f"{drop_percent:.2f}%"
    )

    print(
        f"  Effective FPS:        "
        f"{effective_fps:.2f}"
    )

    print()

    print(
        f"  Predict mean:         "
        f"{result_row['predict_mean_ms']:.2f} ms"
    )

    print(
        f"  Predict median:       "
        f"{result_row['predict_median_ms']:.2f} ms"
    )

    print(
        f"  Predict P95:          "
        f"{result_row['predict_p95_ms']:.2f} ms"
    )

    print(
        f"  Predict P99:          "
        f"{result_row['predict_p99_ms']:.2f} ms"
    )

    print()

    print(
        f"  E2E mean:             "
        f"{result_row['e2e_mean_ms']:.2f} ms"
    )

    print(
        f"  E2E median:           "
        f"{result_row['e2e_median_ms']:.2f} ms"
    )

    print(
        f"  E2E P95:              "
        f"{result_row['e2e_p95_ms']:.2f} ms"
    )

    print(
        f"  E2E P99:              "
        f"{result_row['e2e_p99_ms']:.2f} ms"
    )

    print(
        f"  E2E max:              "
        f"{result_row['e2e_max_ms']:.2f} ms"
    )

    print(
        f"  Deadline misses:      "
        f"{result_row['deadline_miss_percent']:.2f}%"
    )

    print()

    print(
        f"  Ultralytics inference:"
        f" {result_row['ultralytics_inference_mean_ms']:.2f} ms"
    )

    return result_row


def save_summary(
    rows,
):

    LATENCY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "imgsz",

        "source_width",
        "source_height",
        "source_fps",
        "source_frames",
        "frame_period_ms",

        "frames_produced",
        "frames_processed",
        "frames_dropped",
        "drop_percent",

        "effective_fps",

        "predict_mean_ms",
        "predict_median_ms",
        "predict_p95_ms",
        "predict_p99_ms",

        "e2e_mean_ms",
        "e2e_median_ms",
        "e2e_p95_ms",
        "e2e_p99_ms",
        "e2e_max_ms",

        "deadline_miss_percent",

        "ultralytics_inference_mean_ms",
    ]

    with open(
        LATENCY_SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main():

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: "
            f"{VIDEO_PATH}"
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
            f"Could not open video: "
            f"{VIDEO_PATH}"
        )

    ret, warmup_frame = (
        cap.read()
    )

    cap.release()

    if not ret:
        raise RuntimeError(
            "Could not read "
            "warmup frame."
        )

    print("=" * 70)

    print(
        "YOLO26 VIDEO LATENCY "
        "BENCHMARK - 3 CLASS"
    )

    print("=" * 70)

    print(
        f"GPU:       "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"PyTorch:   "
        f"{torch.__version__}"
    )

    print(
        f"ROCm/HIP:  "
        f"{torch.version.hip}"
    )

    print(
        f"Video:     "
        f"{VIDEO_PATH}"
    )

    print(
        f"Models:    "
        f"{MODELS}"
    )

    print(
        f"Sizes:     "
        f"{RESOLUTIONS}"
    )

    print(
        f"Classes:   "
        f"{COCO_EVAL_CLASSES}"
    )

    print(
        f"Viewer:    "
        f"{SHOW_VIEWER}"
    )

    rows = []

    for model_file in MODELS:

        model_name = (
            Path(
                model_file
            ).stem
        )

        print()
        print("=" * 70)

        print(
            f"LOADING "
            f"{model_name}"
        )

        print("=" * 70)

        model = YOLO(
            model_file
        )

        for imgsz in RESOLUTIONS:

            try:

                row = run_test(
                    model_name,
                    model,
                    imgsz,
                    warmup_frame,
                )

                rows.append(
                    row
                )

                save_summary(
                    rows
                )

            except Exception as error:

                print()
                print(
                    f"ERROR: "
                    f"{model_name} "
                    f"@ {imgsz}"
                )

                print(
                    repr(
                        error
                    )
                )

                gc.collect()
                torch.cuda.empty_cache()

        del model

        gc.collect()
        torch.cuda.empty_cache()

    print()
    print("=" * 70)

    print(
        "YOLO26 LATENCY "
        "BENCHMARK COMPLETE"
    )

    print("=" * 70)

    print(
        f"Summary saved to:\n"
        f"{LATENCY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
