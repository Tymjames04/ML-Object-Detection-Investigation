import csv
import gc
import threading
import time

import cv2
import numpy as np
import torch

from torchvision.ops import nms

from experiment_config import (
    MODELS,
    DEVICE,
    WARMUP_RUNS,
    LATENCY_CONF,
    SHOW_VIEWER,
    VIDEO_PATH,
    LATENCY_OUTPUT_DIR,
    TILED_LATENCY_SUMMARY_FILE,
    TARGET_CLASS_NAMES,
    NAME_TO_LOCAL,
    LOCAL_TO_NAME,
    TILE_SIZE,
    TILE_OVERLAP,
    TILED_NMS_IOU,
    TILE_BATCH_SIZE,
)


WINDOW_NAME = (
    "RF-DETR Batched Tiled Latency"
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
    buffer,
    stop_event,
):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        buffer.finish()

        raise RuntimeError(
            f"Could not open video: "
            f"{video_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:

        cap.release()
        buffer.finish()

        raise RuntimeError(
            "Invalid source video FPS."
        )

    period = (
        1.0
        / fps
    )

    start = time.perf_counter()

    buffer.playback_start = (
        start
    )

    frame_id = 0

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            break

        target_time = (
            start
            + frame_id
            * period
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

        buffer.publish(
            frame,
            frame_id,
        )

        frame_id += 1

    cap.release()
    buffer.finish()


def get_starts(
    image_size,
):

    if image_size <= TILE_SIZE:
        return [
            0
        ]

    stride = int(
        TILE_SIZE
        * (
            1.0
            - TILE_OVERLAP
        )
    )

    if stride <= 0:
        raise ValueError(
            "TILE_OVERLAP must be "
            "less than 1.0"
        )

    starts = list(
        range(
            0,
            image_size
            - TILE_SIZE
            + 1,
            stride,
        )
    )

    final_start = (
        image_size
        - TILE_SIZE
    )

    if (
        not starts
        or starts[-1]
        != final_start
    ):
        starts.append(
            final_start
        )

    return starts


def make_tiles(
    frame_bgr,
):

    height, width = (
        frame_bgr.shape[:2]
    )

    tiles_rgb = []
    offsets = []

    for y0 in get_starts(
        height
    ):

        for x0 in get_starts(
            width
        ):

            tile_bgr = frame_bgr[
                y0:
                    min(
                        y0 + TILE_SIZE,
                        height,
                    ),
                x0:
                    min(
                        x0 + TILE_SIZE,
                        width,
                    ),
            ]

            tile_rgb = cv2.cvtColor(
                tile_bgr,
                cv2.COLOR_BGR2RGB,
            )

            tiles_rgb.append(
                tile_rgb
            )

            offsets.append(
                (
                    x0,
                    y0,
                )
            )

    return (
        tiles_rgb,
        offsets,
    )


def batched_tiled_predict(
    frame_bgr,
    model,
):

    frame_height, frame_width = (
        frame_bgr.shape[:2]
    )

    (
        tiles_rgb,
        offsets,
    ) = make_tiles(
        frame_bgr
    )

    all_boxes = []
    all_scores = []
    all_labels = []

    forward_passes = 0

    for batch_start in range(
        0,
        len(tiles_rgb),
        TILE_BATCH_SIZE,
    ):

        batch_end = min(
            batch_start
            + TILE_BATCH_SIZE,
            len(tiles_rgb),
        )

        batch_images = (
            tiles_rgb[
                batch_start:batch_end
            ]
        )

        batch_offsets = (
            offsets[
                batch_start:batch_end
            ]
        )

        detections_list = model.predict(
            batch_images,
            threshold=LATENCY_CONF,
            include_source_image=False,
        )

        forward_passes += 1

        if not isinstance(
            detections_list,
            (list, tuple),
        ):
            raise RuntimeError(
                "RF-DETR batch predict did not "
                "return a detections list."
            )

        if (
            len(detections_list)
            != len(batch_images)
        ):
            raise RuntimeError(
                "RF-DETR batch prediction count "
                "did not match input tile count."
            )

        for (
            detections,
            (
                x0,
                y0,
            ),
        ) in zip(
            detections_list,
            batch_offsets,
        ):

            class_names = (
                detections.data.get(
                    "class_name"
                )
            )

            if class_names is None:

                if len(detections) == 0:
                    continue

                raise RuntimeError(
                    "RF-DETR prediction did "
                    "not contain "
                    "detections.data"
                    "['class_name']."
                )

            class_names = np.asarray(
                class_names,
                dtype=object,
            )

            boxes_np = np.asarray(
                detections.xyxy,
                dtype=np.float32,
            )

            scores_np = np.asarray(
                detections.confidence,
                dtype=np.float32,
            )

            for (
                box,
                score,
                class_name,
            ) in zip(
                boxes_np,
                scores_np,
                class_names,
            ):

                class_name = str(
                    class_name
                )

                if (
                    class_name
                    not in TARGET_CLASS_NAMES
                ):
                    continue

                global_box = torch.tensor(
                    box,
                    dtype=torch.float32,
                )

                global_box[0] += x0
                global_box[2] += x0
                global_box[1] += y0
                global_box[3] += y0

                global_box[0].clamp_(
                    0,
                    frame_width,
                )

                global_box[2].clamp_(
                    0,
                    frame_width,
                )

                global_box[1].clamp_(
                    0,
                    frame_height,
                )

                global_box[3].clamp_(
                    0,
                    frame_height,
                )

                if (
                    global_box[2]
                    <= global_box[0]
                    or global_box[3]
                    <= global_box[1]
                ):
                    continue

                all_boxes.append(
                    global_box
                )

                all_scores.append(
                    float(
                        score
                    )
                )

                all_labels.append(
                    NAME_TO_LOCAL[
                        class_name
                    ]
                )

    if not all_boxes:
        return {
            "boxes":
                torch.empty(
                    (0, 4),
                    dtype=torch.float32,
                ),

            "scores":
                torch.empty(
                    (0,),
                    dtype=torch.float32,
                ),

            "labels":
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                ),

            "tiles":
                len(
                    tiles_rgb
                ),

            "forward_passes":
                forward_passes,
        }

    boxes = torch.stack(
        all_boxes
    )

    scores = torch.tensor(
        all_scores,
        dtype=torch.float32,
    )

    labels = torch.tensor(
        all_labels,
        dtype=torch.int64,
    )

    kept_indices = []

    for class_id in torch.unique(
        labels
    ):

        class_indices = torch.where(
            labels == class_id
        )[0]

        keep = nms(
            boxes[
                class_indices
            ],
            scores[
                class_indices
            ],
            TILED_NMS_IOU,
        )

        kept_indices.append(
            class_indices[
                keep
            ]
        )

    if kept_indices:

        keep = torch.cat(
            kept_indices
        )

        keep = keep[
            torch.argsort(
                scores[
                    keep
                ],
                descending=True,
            )
        ]

        boxes = boxes[
            keep
        ]

        scores = scores[
            keep
        ]

        labels = labels[
            keep
        ]

    return {
        "boxes":
            boxes
            .detach()
            .cpu(),

        "scores":
            scores
            .detach()
            .cpu(),

        "labels":
            labels
            .detach()
            .cpu(),

        "tiles":
            len(
                tiles_rgb
            ),

        "forward_passes":
            forward_passes,
    }


def draw_predictions(
    frame_bgr,
    prediction,
):

    frame = (
        frame_bgr.copy()
    )

    boxes = (
        prediction[
            "boxes"
        ].numpy()
    )

    scores = (
        prediction[
            "scores"
        ].numpy()
    )

    labels = (
        prediction[
            "labels"
        ].numpy()
    )

    for (
        box,
        score,
        label,
    ) in zip(
        boxes,
        scores,
        labels,
    ):

        x1, y1, x2, y2 = (
            box.astype(
                int
            )
        )

        name = (
            LOCAL_TO_NAME[
                int(label)
            ]
        )

        cv2.rectangle(
            frame,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

        cv2.putText(
            frame,
            (
                f"{name} "
                f"{score:.2f}"
            ),
            (
                x1,
                max(
                    15,
                    y1 - 5,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (
                0,
                255,
                0,
            ),
            1,
            cv2.LINE_AA,
        )

    return frame


def warmup(
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
        f"  Warmup:          "
        f"{WARMUP_RUNS} frames"
    )

    print(
        "  Warmup..."
    )

    while (
        completed
        < WARMUP_RUNS
    ):

        ret, frame = cap.read()

        if not ret:
            break

        _ = batched_tiled_predict(
            frame,
            model,
        )

        completed += 1

    cap.release()

    torch.cuda.synchronize()


def run_test(
    model_name,
    model_class,
):

    print()
    print(
        f"LOADING "
        f"{model_name}"
    )

    model = model_class(
        resolution=TILE_SIZE,
        device=DEVICE,
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

    tiles_per_frame = (
        len(
            get_starts(
                source_width
            )
        )
        *
        len(
            get_starts(
                source_height
            )
        )
    )

    forward_passes_per_frame = int(
        np.ceil(
            tiles_per_frame
            / TILE_BATCH_SIZE
        )
    )

    print()
    print("=" * 70)

    print(
        f"{model_name.upper()} "
        f"BATCHED TILED "
        f"{TILE_SIZE}x"
        f"{TILE_SIZE}"
    )

    print("=" * 70)

    print(
        f"  Source:          "
        f"{source_width}x"
        f"{source_height} "
        f"@ {source_fps:.3f} FPS"
    )

    print(
        f"  Tile size:       "
        f"{TILE_SIZE}"
    )

    print(
        f"  Overlap:         "
        f"{TILE_OVERLAP:.0%}"
    )

    print(
        f"  Tiles/frame:     "
        f"{tiles_per_frame}"
    )

    print(
        f"  Tile batch size: "
        f"{TILE_BATCH_SIZE}"
    )

    print(
        f"  Forward passes:  "
        f"{forward_passes_per_frame}/frame"
    )

    print(
        f"  Confidence:      "
        f"{LATENCY_CONF}"
    )

    print(
        f"  NMS IoU:         "
        f"{TILED_NMS_IOU}"
    )

    warmup(
        model
    )

    buffer = LatestFrameBuffer()
    stop_event = (
        threading.Event()
    )

    producer = threading.Thread(
        target=video_producer,
        args=(
            VIDEO_PATH,
            buffer,
            stop_event,
        ),
        daemon=True,
    )

    producer.start()

    while (
        buffer.playback_start
        is None
        and producer.is_alive()
    ):
        time.sleep(
            0.001
        )

    if (
        buffer.playback_start
        is None
    ):
        raise RuntimeError(
            "Video producer failed "
            "to start."
        )

    run_start = (
        buffer.playback_start
    )

    predict_times = []
    e2e_times = []
    detection_counts = []

    processed = 0
    last_frame_id = -1

    while True:

        item = buffer.get_latest(
            last_frame_id
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

        # Full batched tiled pipeline:
        # tile creation -> BGR/RGB -> RF-DETR batch predict
        # -> coordinate remap -> class filtering -> NMS
        torch.cuda.synchronize()

        predict_start = (
            time.perf_counter()
        )

        prediction = (
            batched_tiled_predict(
                frame_bgr,
                model,
            )
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

        detection_counts.append(
            len(
                prediction[
                    "boxes"
                ]
            )
        )

        key = -1

        if SHOW_VIEWER:

            annotated = (
                draw_predictions(
                    frame_bgr,
                    prediction,
                )
            )

            cv2.putText(
                annotated,
                (
                    f"{model_name} | "
                    f"{prediction['tiles']} tiles | "
                    f"{prediction['forward_passes']} forward | "
                    f"{predict_ms:.1f} ms"
                ),
                (
                    20,
                    30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
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
            processed % 100
            == 0
        ):

            print(
                f"  Processed "
                f"{processed} | "
                f"Predict "
                f"{predict_ms:.2f} ms | "
                f"E2E "
                f"{e2e_ms:.2f} ms | "
                f"Detections "
                f"{detection_counts[-1]}"
            )

        if key == ord(
            "q"
        ):

            stop_event.set()

            break

    stop_event.set()
    producer.join()

    run_end = (
        time.perf_counter()
    )

    if SHOW_VIEWER:
        cv2.destroyAllWindows()

    produced = (
        buffer.produced
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

    e2e_times = np.asarray(
        e2e_times,
        dtype=np.float64,
    )

    detection_counts = np.asarray(
        detection_counts,
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

    result = {
        "model":
            model_name,

        "method":
            "tiled_batched",

        "tile_size":
            TILE_SIZE,

        "overlap":
            TILE_OVERLAP,

        "nms_iou":
            TILED_NMS_IOU,

        "confidence":
            LATENCY_CONF,

        "tiles_per_frame":
            tiles_per_frame,

        "tile_batch_size":
            TILE_BATCH_SIZE,

        "forward_passes_per_frame":
            forward_passes_per_frame,

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

        "mean_detections_per_frame":
            float(
                np.mean(
                    detection_counts
                )
            ),
    }

    print()
    print(
        f"  Predict mean: "
        f"{result['predict_mean_ms']:.2f} ms"
    )

    print(
        f"  E2E mean:     "
        f"{result['e2e_mean_ms']:.2f} ms"
    )

    print(
        f"  Effective FPS: "
        f"{result['effective_fps']:.2f}"
    )

    print(
        f"  Drop rate:     "
        f"{result['drop_percent']:.2f}%"
    )

    print(
        f"  Deadline miss: "
        f"{result['deadline_miss_percent']:.2f}%"
    )

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
        "method",

        "tile_size",
        "overlap",
        "nms_iou",
        "confidence",

        "tiles_per_frame",
        "tile_batch_size",
        "forward_passes_per_frame",

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
        "mean_detections_per_frame",
    ]

    with open(
        TILED_LATENCY_SUMMARY_FILE,
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

    print("=" * 70)

    print(
        "RF-DETR BATCHED TILED "
        "FPV LATENCY BENCHMARK"
    )

    print("=" * 70)

    print(
        f"GPU:             "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Models:          "
        f"{list(MODELS.keys())}"
    )

    print(
        f"Tile size:       "
        f"{TILE_SIZE}"
    )

    print(
        f"Overlap:         "
        f"{TILE_OVERLAP:.0%}"
    )

    print(
        f"Tile batch size: "
        f"{TILE_BATCH_SIZE}"
    )

    print(
        f"NMS IoU:         "
        f"{TILED_NMS_IOU}"
    )

    print(
        f"Confidence:      "
        f"{LATENCY_CONF}"
    )

    print(
        f"Classes:         "
        f"{TARGET_CLASS_NAMES}"
    )

    print(
        f"Viewer:          "
        f"{SHOW_VIEWER}"
    )

    rows = []

    for (
        model_name,
        model_class,
    ) in MODELS.items():

        try:

            row = run_test(
                model_name,
                model_class,
            )

            rows.append(
                row
            )

            save_summary(
                rows
            )

        except torch.cuda.OutOfMemoryError:

            print()
            print(
                f"OUT OF VRAM: "
                f"{model_name}"
            )

            print(
                "Change TILE_BATCH_SIZE "
                f"from {TILE_BATCH_SIZE} "
                "to 4 and rerun."
            )

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as error:

            print()
            print(
                f"ERROR: "
                f"{model_name}"
            )

            print(
                repr(
                    error
                )
            )

            gc.collect()
            torch.cuda.empty_cache()

    print()
    print("=" * 70)

    print(
        "RF-DETR BATCHED TILED "
        "LATENCY COMPLETE"
    )

    print("=" * 70)

    print(
        f"Summary saved to:\n"
        f"{TILED_LATENCY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
