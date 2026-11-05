import csv
import gc
import threading
import time

import cv2
import numpy as np
import torch

from transformers import (
    RTDetrImageProcessor,
    RTDetrV2ForObjectDetection,
)

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
    MODEL_NAME_ALIASES,
)


# ============================================================
# LATEST-FRAME BUFFER
# ============================================================



class LatestFrameBuffer:

    def __init__(self):

        self.condition = threading.Condition()

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
            self.arrival_time = time.perf_counter()

            self.produced += 1

            self.condition.notify_all()

    def finish(self):

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
                    self.frame_id > last_frame_id
                    or self.finished
            )

            if self.frame_id > last_frame_id:

                return (
                    self.frame_id,
                    self.frame,
                    self.arrival_time,
                )

            return None


# ============================================================
# CLASS-NAME NORMALIZATION
# ============================================================

def normalize_class_name(
    class_name,
):

    class_name = str(
        class_name
    ).lower()

    return MODEL_NAME_ALIASES.get(
        class_name,
        class_name,
    )


# ============================================================
# FILTER TARGET CLASSES
# ============================================================

def filter_target_classes(
    prediction,
    model,
):

    boxes = (
        prediction["boxes"]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        prediction["scores"]
        .detach()
        .cpu()
        .numpy()
    )

    labels = (
        prediction["labels"]
        .detach()
        .cpu()
        .numpy()
    )

    filtered = []

    for (
        box,
        score,
        label_id,
    ) in zip(
        boxes,
        scores,
        labels,
    ):

        label_id = int(
            label_id
        )

        raw_name = (
            model.config.id2label[
                label_id
            ]
        )

        class_name = (
            normalize_class_name(
                raw_name
            )
        )

        if (
            class_name
            not in TARGET_CLASS_NAMES
        ):
            continue

        filtered.append(
            (
                box,
                float(score),
                class_name,
            )
        )

    return filtered


# ============================================================
# VIDEO PRODUCER
# ============================================================

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
            "Invalid source FPS."
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

        # Replay at the video's native FPS.
        target_time = (
            playback_start
            + frame_id
            * frame_period
        )

        now = time.perf_counter()

        sleep_time = (
            target_time - now
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


# ============================================================
# PREPROCESS + FORWARD + POSTPROCESS
# ============================================================

def predict_frame(
    frame_rgb,
    processor,
    model,
    resolution,
    threshold,
):

    image_height, image_width = (
        frame_rgb.shape[:2]
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    inputs = processor(
        images=frame_rgb,

        size={
            "height":
                resolution,

            "width":
                resolution,
        },

        return_tensors="pt",
    )

    inputs = {
        key:
            value.to(
                DEVICE
            )

        for (
            key,
            value,
        ) in inputs.items()
    }

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model(
            **inputs
        )

    # --------------------------------------------------------
    # POSTPROCESS
    #
    # Convert normalized RT-DETR output back to absolute
    # source-image xyxy coordinates.
    # --------------------------------------------------------

    target_sizes = torch.tensor(
        [
            [
                image_height,
                image_width,
            ]
        ],
        device=DEVICE,
    )

    prediction = (
        processor
        .post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )[0]
    )

    return prediction


# ============================================================
# WARMUP
# ============================================================

def warmup_model(
    processor,
    model,
    resolution,
):

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video "
            f"for warmup: {VIDEO_PATH}"
        )

    completed = 0

    print(
        f"  Warming up with "
        f"{WARMUP_RUNS} frames..."
    )

    while completed < WARMUP_RUNS:

        ret, frame_bgr = cap.read()

        if not ret:
            break

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        _ = predict_frame(
            frame_rgb,
            processor,
            model,
            resolution,
            LATENCY_CONF,
        )

        completed += 1

    cap.release()

    torch.cuda.synchronize()

    print(
        "  Warmup complete."
    )


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    frame,
    detections,
):

    annotated = (
        frame.copy()
    )

    for (
        box,
        confidence,
        class_name,
    ) in detections:

        x1, y1, x2, y2 = (
            box.astype(int)
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        text = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            annotated,
            text,
            (
                x1,
                max(
                    y1 - 5,
                    15,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated


# ============================================================
# RUN ONE RESOLUTION
# ============================================================

def run_test(
    model_name,
    processor,
    model,
    resolution,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{model_name.upper()} "
        f"@ {resolution}x{resolution}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # WARMUP
    # --------------------------------------------------------

    warmup_model(
        processor,
        model,
        resolution,
    )

    # --------------------------------------------------------
    # SOURCE VIDEO INFORMATION
    # --------------------------------------------------------

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

    frame_period_ms = (
        1000.0
        / source_fps
    )

    print(
        f"  Source resolution: "
        f"{source_width}x{source_height}"
    )

    print(
        f"  Source FPS:        "
        f"{source_fps:.2f}"
    )

    print(
        f"  Frame period:      "
        f"{frame_period_ms:.2f} ms"
    )

    # --------------------------------------------------------
    # PRODUCER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

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

    # ========================================================
    # CONSUMER
    # ========================================================

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

        # ----------------------------------------------------
        # BGR -> RGB
        #
        # Kept OUTSIDE predict timing but INSIDE E2E,
        # matching the RF-DETR methodology.
        # ----------------------------------------------------

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        # ----------------------------------------------------
        # RT-DETRv2 PREDICT LATENCY
        #
        # Includes:
        #   image processor
        #   CPU -> GPU transfer
        #   network forward
        #   RT-DETR postprocessing
        #   conversion to source-image boxes
        #
        # Does NOT include:
        #   class filtering
        #   drawing
        #   display
        # ----------------------------------------------------

        torch.cuda.synchronize()

        predict_start = (
            time.perf_counter()
        )

        prediction = predict_frame(
            frame_rgb,
            processor,
            model,
            resolution,
            LATENCY_CONF,
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

        # ----------------------------------------------------
        # TARGET CLASS FILTER
        # ----------------------------------------------------

        detections = (
            filter_target_classes(
                prediction,
                model,
            )
        )

        # ----------------------------------------------------
        # VIEWER
        # ----------------------------------------------------

        key = -1

        if SHOW_VIEWER:

            annotated = (
                draw_detections(
                    frame_bgr,
                    detections,
                )
            )

            cv2.imshow(
                (
                    f"{WINDOW_NAME} - "
                    f"{model_name} "
                    f"@ {resolution}"
                ),
                annotated,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

        # ----------------------------------------------------
        # END-TO-END LATENCY
        # ----------------------------------------------------

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

        if key == ord("q"):

            stop_event.set()

            break

    # --------------------------------------------------------
    # CLEANUP PRODUCER
    # --------------------------------------------------------

    stop_event.set()

    producer_thread.join()

    run_end = (
        time.perf_counter()
    )

    if SHOW_VIEWER:

        cv2.destroyAllWindows()

    # --------------------------------------------------------
    # FRAME COUNTS
    # --------------------------------------------------------

    produced = (
        frame_buffer.produced
    )

    dropped = max(
        produced
        - processed,
        0,
    )

    # --------------------------------------------------------
    # NUMPY
    # --------------------------------------------------------

    predict_times = np.asarray(
        predict_times,
        dtype=np.float64,
    )

    e2e_times = np.asarray(
        e2e_times,
        dtype=np.float64,
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

    # --------------------------------------------------------
    # PREDICT METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # E2E METRICS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print(
        "  Results"
    )
    print(
        "  -------"
    )

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
        f"  Predict median:       "
        f"{predict_median:.2f} ms"
    )

    print(
        f"  Predict P95:          "
        f"{predict_p95:.2f} ms"
    )

    print(
        f"  Predict P99:          "
        f"{predict_p99:.2f} ms"
    )

    print()

    print(
        f"  E2E mean:             "
        f"{e2e_mean:.2f} ms"
    )

    print(
        f"  E2E median:           "
        f"{e2e_median:.2f} ms"
    )

    print(
        f"  E2E P95:              "
        f"{e2e_p95:.2f} ms"
    )

    print(
        f"  E2E P99:              "
        f"{e2e_p99:.2f} ms"
    )

    print(
        f"  E2E max:              "
        f"{e2e_max:.2f} ms"
    )

    print(
        f"  Deadline misses:      "
        f"{deadline_miss_percent:.2f}%"
    )

    return {
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


# ============================================================
# SAVE CSV
# ============================================================

def save_summary(
    rows
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


# ============================================================
# MAIN
# ============================================================

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

    print(
        "=" * 70
    )

    print(
        "RT-DETRv2 VIDEO LATENCY BENCHMARK - 3 CLASS"
    )

    print(
        "=" * 70
    )

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

    # ========================================================
    # MODELS
    # ========================================================

    for (
        model_name,
        model_id,
    ) in MODELS.items():

        print()
        print(
            "=" * 70
        )

        print(
            f"LOADING "
            f"{model_name}"
        )

        print(
            "=" * 70
        )

        processor = (
            RTDetrImageProcessor
            .from_pretrained(
                model_id
            )
        )

        model = (
            RTDetrV2ForObjectDetection
            .from_pretrained(
                model_id
            )
            .to(
                DEVICE
            )
            .eval()
        )

        for resolution in RESOLUTIONS:

            try:

                row = run_test(
                    model_name,
                    processor,
                    model,
                    resolution,
                )

                results.append(
                    row
                )

                # Save immediately after each successful run.
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

        del model
        del processor

        gc.collect()

        torch.cuda.empty_cache()

    print()
    print(
        "=" * 70
    )

    print(
        "LATENCY BENCHMARK COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Summary saved to:\n"
        f"{LATENCY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()