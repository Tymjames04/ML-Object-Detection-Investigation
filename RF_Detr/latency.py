import csv
import gc
import threading
import time

import cv2
import numpy as np
import supervision as sv
import torch

from experiment_config import (
    MODELS,
    RESOLUTIONS,
    DEVICE,
    WARMUP_RUNS,
    LATENCY_CONF,
    SHOW_VIEWER,
    WINDOW_NAME,
    VIDEO_PATH,
    LATENCY_OUTPUT_DIR,
    LATENCY_SUMMARY_FILE,
    TARGET_CLASS_NAMES,
)


class LatestFrameBuffer:

    def __init__(self):

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


def filter_target_classes(
    detections,
):

    class_names = (
        detections.data.get(
            "class_name"
        )
    )

    if class_names is None:

        if len(detections) == 0:
            return detections

        raise RuntimeError(
            "RF-DETR prediction did not "
            "contain "
            "detections.data['class_name']."
        )

    class_names = (
        np.asarray(
            class_names,
            dtype=object,
        )
    )

    class_mask = np.isin(
        class_names,
        TARGET_CLASS_NAMES,
    )

    return (
        detections[
            class_mask
        ]
    )


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
        1.0 / source_fps
    )

    frame_id = 0

    playback_start = (
        time.perf_counter()
    )

    frame_buffer.playback_start = (
        playback_start
    )

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            break

        target_time = (
            playback_start
            + frame_id
            * frame_period
        )

        sleep_time = (
            target_time
            - time.perf_counter()
        )

        if sleep_time > 0:

            time.sleep(
                sleep_time
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
):

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video "
            f"for warmup: "
            f"{VIDEO_PATH}"
        )

    completed = 0

    print(
        f"  Warming up with "
        f"{WARMUP_RUNS} frames..."
    )

    while (
        completed
        < WARMUP_RUNS
    ):

        ret, frame_bgr = (
            cap.read()
        )

        if not ret:
            break

        frame_rgb = (
            cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )
        )

        model.predict(
            frame_rgb,
            threshold=LATENCY_CONF,
            include_source_image=False,
        )

        completed += 1

    cap.release()

    torch.cuda.synchronize()

    print(
        "  Warmup complete."
    )


def run_test(
    model_name,
    model_class,
    resolution,
):

    print()
    print("=" * 65)

    print(
        f"{model_name.upper()} @ "
        f"{resolution}x{resolution}"
    )

    print("=" * 65)

    print(
        "  Loading model..."
    )

    model = model_class(
        resolution=resolution,
        device=DEVICE,
    )

    warmup_model(
        model
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
        f"{source_fps:.2f}"
    )

    print(
        f"  Source frames:     "
        f"{source_frames}"
    )

    print(
        f"  Frame period:      "
        f"{frame_period_ms:.2f} ms"
    )

    if SHOW_VIEWER:

        box_annotator = (
            sv.BoxAnnotator()
        )

        label_annotator = (
            sv.LabelAnnotator()
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

    predict_times = []
    e2e_times = []

    processed = 0
    last_frame_id = -1

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

        # Kept outside predict timing,
        # but inside E2E.
        frame_rgb = (
            cv2.cvtColor(
                frame_bgr,
                cv2.COLOR_BGR2RGB,
            )
        )

        torch.cuda.synchronize()

        predict_start = (
            time.perf_counter()
        )

        detections = model.predict(
            frame_rgb,
            threshold=LATENCY_CONF,
            include_source_image=False,
        )

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

        detections = (
            filter_target_classes(
                detections
            )
        )

        if SHOW_VIEWER:

            class_names = (
                detections.data[
                    "class_name"
                ]
            )

            labels = [
                (
                    f"{class_name} "
                    f"{confidence:.2f}"
                )
                for (
                    class_name,
                    confidence,
                )
                in zip(
                    class_names,
                    detections.confidence,
                )
            ]

            annotated_frame = (
                box_annotator.annotate(
                    scene=frame_bgr.copy(),
                    detections=detections,
                )
            )

            annotated_frame = (
                label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels,
                )
            )

            cv2.imshow(
                WINDOW_NAME,
                annotated_frame,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):

                stop_event.set()

                break

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

    predict_times = (
        np.asarray(
            predict_times,
            dtype=np.float64,
        )
    )

    e2e_times = (
        np.asarray(
            e2e_times,
            dtype=np.float64,
        )
    )

    if processed == 0:

        raise RuntimeError(
            "No frames were processed."
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

    predict_mean = float(
        np.mean(
            predict_times
        )
    )

    predict_median = float(
        np.median(
            predict_times
        )
    )

    predict_p95 = float(
        np.percentile(
            predict_times,
            95,
        )
    )

    predict_p99 = float(
        np.percentile(
            predict_times,
            99,
        )
    )

    e2e_mean = float(
        np.mean(
            e2e_times
        )
    )

    e2e_median = float(
        np.median(
            e2e_times
        )
    )

    e2e_p95 = float(
        np.percentile(
            e2e_times,
            95,
        )
    )

    e2e_p99 = float(
        np.percentile(
            e2e_times,
            99,
        )
    )

    e2e_max = float(
        np.max(
            e2e_times
        )
    )

    deadline_miss_percent = float(
        100.0
        * np.mean(
            e2e_times
            > frame_period_ms
        )
    )

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
        f"{predict_mean:.2f} ms"
    )

    print(
        f"  E2E mean:             "
        f"{e2e_mean:.2f} ms"
    )

    print(
        f"  Deadline misses:      "
        f"{deadline_miss_percent:.2f}%"
    )

    result = {
        "model":
            model_name,

        "imgsz":
            resolution,

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
            predict_mean,

        "predict_median_ms":
            predict_median,

        "predict_p95_ms":
            predict_p95,

        "predict_p99_ms":
            predict_p99,

        "e2e_mean_ms":
            e2e_mean,

        "e2e_median_ms":
            e2e_median,

        "e2e_p95_ms":
            e2e_p95,

        "e2e_p99_ms":
            e2e_p99,

        "e2e_max_ms":
            e2e_max,

        "deadline_miss_percent":
            deadline_miss_percent,
    }

    del model

    gc.collect()

    torch.cuda.empty_cache()

    return result


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

    print("=" * 65)

    print(
        "RF-DETR VIDEO LATENCY "
        "BENCHMARK - 3 CLASS"
    )

    print("=" * 65)

    print(
        f"GPU:       "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Video:     "
        f"{VIDEO_PATH}"
    )

    print(
        f"Models:    "
        f"{list(MODELS.keys())}"
    )

    print(
        f"Sizes:     "
        f"{RESOLUTIONS}"
    )

    print(
        f"Classes:   "
        f"{TARGET_CLASS_NAMES}"
    )

    print(
        f"Viewer:    "
        f"{SHOW_VIEWER}"
    )

    results = []

    for (
        model_name,
        model_class,
    ) in MODELS.items():

        for resolution in RESOLUTIONS:

            try:

                row = run_test(
                    model_name,
                    model_class,
                    resolution,
                )

                results.append(
                    row
                )

                save_summary(
                    results
                )

            except Exception as error:

                print()

                print(
                    f"ERROR: "
                    f"{model_name} "
                    f"@ {resolution}"
                )

                print(
                    repr(
                        error
                    )
                )

                gc.collect()

                torch.cuda.empty_cache()

    print()
    print("=" * 65)

    print(
        "BENCHMARK COMPLETE"
    )

    print("=" * 65)

    print(
        f"Summary saved to:\n"
        f"{LATENCY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
