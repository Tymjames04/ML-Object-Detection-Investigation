import csv
import gc
import threading
import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from experiment_config import (
    MODELS,
    DEVICE,
    WARMUP_RUNS,
    LATENCY_CONF,
    SHOW_VIEWER,
    VIDEO_PATH,
    LATENCY_OUTPUT_DIR,
    TARGET_CLASS_NAMES,
    MODEL_NAME_ALIASES,
    NAME_TO_LOCAL,
    TILE_SIZE,
    TILE_OVERLAP,
    TILED_NMS_IOU,
    TILE_BATCH_SIZE,
    TILED_LATENCY_SUMMARY_FILE,
)

WINDOW_NAME = "RT-DETRv2 Batched Tiled Latency"


class LatestFrameBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = -1
        self.arrival_time = None
        self.produced = 0
        self.finished = False
        self.playback_start = None

    def publish(self, frame, frame_id):
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

    def get_latest(self, last_frame_id):
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_id > last_frame_id or self.finished
            )
            if self.frame_id > last_frame_id:
                return self.frame_id, self.frame, self.arrival_time
            return None


def video_producer(video_path, buffer, stop_event):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        buffer.finish()
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        buffer.finish()
        raise RuntimeError("Invalid source video FPS.")

    period = 1.0 / fps
    start = time.perf_counter()
    buffer.playback_start = start
    frame_id = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        target_time = start + frame_id * period
        delay = target_time - time.perf_counter()

        if delay > 0:
            time.sleep(delay)

        if stop_event.is_set():
            break

        buffer.publish(frame, frame_id)
        frame_id += 1

    cap.release()
    buffer.finish()


def normalize_name(name):
    name = str(name).lower()
    return MODEL_NAME_ALIASES.get(name, name)


def get_starts(image_size):
    if image_size <= TILE_SIZE:
        return [0]

    stride = int(TILE_SIZE * (1.0 - TILE_OVERLAP))
    if stride <= 0:
        raise ValueError("TILE_OVERLAP must be less than 1.0")

    starts = list(range(0, image_size - TILE_SIZE + 1, stride))
    final_start = image_size - TILE_SIZE

    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return starts


def make_tiles(frame_bgr):
    height, width = frame_bgr.shape[:2]

    tiles_rgb = []
    offsets = []
    sizes = []

    for y0 in get_starts(height):
        for x0 in get_starts(width):
            tile_bgr = frame_bgr[
                y0:y0 + TILE_SIZE,
                x0:x0 + TILE_SIZE,
            ]

            tile_rgb = cv2.cvtColor(
                tile_bgr,
                cv2.COLOR_BGR2RGB,
            )

            tile_h, tile_w = tile_rgb.shape[:2]

            tiles_rgb.append(tile_rgb)
            offsets.append((x0, y0))
            sizes.append((tile_h, tile_w))

    return tiles_rgb, offsets, sizes


def build_class_map(model):
    mapping = {}

    for label_id, raw_name in model.config.id2label.items():
        name = normalize_name(raw_name)

        if name in TARGET_CLASS_NAMES:
            mapping[int(label_id)] = NAME_TO_LOCAL[name]

    if not mapping:
        raise RuntimeError(
            "No target classes matched model.config.id2label."
        )

    return mapping


def batched_tiled_predict(
    frame_bgr,
    processor,
    model,
    class_map,
):
    frame_h, frame_w = frame_bgr.shape[:2]

    tiles, offsets, sizes = make_tiles(frame_bgr)

    all_boxes = []
    all_scores = []
    all_labels = []

    forward_passes = 0

    for start in range(0, len(tiles), TILE_BATCH_SIZE):
        end = min(start + TILE_BATCH_SIZE, len(tiles))

        batch_images = tiles[start:end]
        batch_offsets = offsets[start:end]
        batch_sizes = sizes[start:end]

        # One preprocessing call for this whole tile batch.
        inputs = processor(
            images=batch_images,
            size={
                "height": TILE_SIZE,
                "width": TILE_SIZE,
            },
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # One model forward for this whole tile batch.
        with torch.inference_mode():
            outputs = model(**inputs)

        forward_passes += 1

        target_sizes = torch.tensor(
            batch_sizes,
            dtype=torch.int64,
            device=DEVICE,
        )

        predictions = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=LATENCY_CONF,
        )

        for prediction, (x0, y0) in zip(
            predictions,
            batch_offsets,
        ):
            boxes = prediction["boxes"].to(torch.float32)
            scores = prediction["scores"].to(torch.float32)
            model_labels = prediction["labels"].to(torch.int64)

            for model_label_id, local_class_id in class_map.items():
                mask = model_labels == model_label_id

                if not torch.any(mask):
                    continue

                class_boxes = boxes[mask].clone()
                class_scores = scores[mask]

                class_boxes[:, 0] += x0
                class_boxes[:, 2] += x0
                class_boxes[:, 1] += y0
                class_boxes[:, 3] += y0

                class_boxes[:, 0].clamp_(0, frame_w)
                class_boxes[:, 2].clamp_(0, frame_w)
                class_boxes[:, 1].clamp_(0, frame_h)
                class_boxes[:, 3].clamp_(0, frame_h)

                valid = (
                    (class_boxes[:, 2] > class_boxes[:, 0])
                    &
                    (class_boxes[:, 3] > class_boxes[:, 1])
                )

                class_boxes = class_boxes[valid]
                class_scores = class_scores[valid]

                if class_boxes.shape[0] == 0:
                    continue

                class_labels = torch.full(
                    (class_boxes.shape[0],),
                    local_class_id,
                    dtype=torch.int64,
                    device=class_boxes.device,
                )

                all_boxes.append(class_boxes)
                all_scores.append(class_scores)
                all_labels.append(class_labels)

    if not all_boxes:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
            "tiles": len(tiles),
            "forward_passes": forward_passes,
        }

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    kept = []

    # Class-aware NMS.
    for class_id in torch.unique(labels):
        class_indices = torch.where(labels == class_id)[0]

        class_keep = nms(
            boxes[class_indices],
            scores[class_indices],
            TILED_NMS_IOU,
        )

        kept.append(class_indices[class_keep])

    if kept:
        keep = torch.cat(kept)

        keep = keep[
            torch.argsort(
                scores[keep],
                descending=True,
            )
        ]

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    return {
        "boxes": boxes.detach().cpu(),
        "scores": scores.detach().cpu(),
        "labels": labels.detach().cpu(),
        "tiles": len(tiles),
        "forward_passes": forward_passes,
    }


def draw_predictions(frame_bgr, prediction):
    frame = frame_bgr.copy()

    local_to_name = {
        local_id: name
        for name, local_id in NAME_TO_LOCAL.items()
    }

    for box, score, label in zip(
        prediction["boxes"].numpy(),
        prediction["scores"].numpy(),
        prediction["labels"].numpy(),
    ):
        x1, y1, x2, y2 = box.astype(int)
        name = local_to_name[int(label)]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return frame


def warmup(processor, model, class_map):
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video for warmup: {VIDEO_PATH}"
        )

    completed = 0

    print(f"  Warmup:          {WARMUP_RUNS} frames")
    print("  Warmup...")

    while completed < WARMUP_RUNS:
        ret, frame = cap.read()

        if not ret:
            break

        _ = batched_tiled_predict(
            frame,
            processor,
            model,
            class_map,
        )

        completed += 1

    cap.release()
    torch.cuda.synchronize()


def run_test(model_name, model_id):
    print()
    print(f"LOADING {model_name}")

    processor = RTDetrImageProcessor.from_pretrained(
        model_id
    )

    model = (
        RTDetrV2ForObjectDetection
        .from_pretrained(model_id)
        .to(DEVICE)
        .eval()
    )

    class_map = build_class_map(model)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if source_fps <= 0:
        raise RuntimeError("Invalid source FPS.")

    frame_period_ms = 1000.0 / source_fps

    tiles_per_frame = (
        len(get_starts(source_width))
        * len(get_starts(source_height))
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
        f"BATCHED TILED {TILE_SIZE}x{TILE_SIZE}"
    )
    print("=" * 70)

    print(
        f"  Source:          "
        f"{source_width}x{source_height} "
        f"@ {source_fps:.3f} FPS"
    )
    print(f"  Tile size:       {TILE_SIZE}")
    print(f"  Overlap:         {TILE_OVERLAP:.0%}")
    print(f"  Tiles/frame:     {tiles_per_frame}")
    print(f"  Tile batch size: {TILE_BATCH_SIZE}")
    print(
        f"  Forward passes:  "
        f"{forward_passes_per_frame}/frame"
    )
    print(f"  Confidence:      {LATENCY_CONF}")
    print(f"  NMS IoU:         {TILED_NMS_IOU}")

    warmup(
        processor,
        model,
        class_map,
    )

    buffer = LatestFrameBuffer()
    stop_event = threading.Event()

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
        buffer.playback_start is None
        and producer.is_alive()
    ):
        time.sleep(0.001)

    if buffer.playback_start is None:
        raise RuntimeError(
            "Video producer failed to start."
        )

    run_start = buffer.playback_start

    predict_times = []
    e2e_times = []
    detection_counts = []

    processed = 0
    last_frame_id = -1

    while True:
        item = buffer.get_latest(last_frame_id)

        if item is None:
            break

        frame_id, frame_bgr, arrival_time = item
        last_frame_id = frame_id

        # Full batched tiled pipeline:
        # tile creation -> preprocessing -> H2D -> batched forward
        # -> postprocess -> coordinate remap -> class filtering -> NMS
        torch.cuda.synchronize()
        predict_start = time.perf_counter()

        prediction = batched_tiled_predict(
            frame_bgr,
            processor,
            model,
            class_map,
        )

        torch.cuda.synchronize()
        predict_end = time.perf_counter()

        predict_ms = (
            predict_end - predict_start
        ) * 1000.0

        predict_times.append(predict_ms)
        detection_counts.append(
            len(prediction["boxes"])
        )

        key = -1

        if SHOW_VIEWER:
            annotated = draw_predictions(
                frame_bgr,
                prediction,
            )

            cv2.putText(
                annotated,
                (
                    f"{model_name} | "
                    f"{prediction['tiles']} tiles | "
                    f"{prediction['forward_passes']} forward | "
                    f"{predict_ms:.1f} ms"
                ),
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                WINDOW_NAME,
                annotated,
            )

            key = cv2.waitKey(1) & 0xFF

        completion_time = time.perf_counter()

        e2e_ms = (
            completion_time - arrival_time
        ) * 1000.0

        e2e_times.append(e2e_ms)
        processed += 1

        if processed % 100 == 0:
            print(
                f"  Processed {processed} | "
                f"Predict {predict_ms:.2f} ms | "
                f"E2E {e2e_ms:.2f} ms | "
                f"Detections {detection_counts[-1]}"
            )

        if key == ord("q"):
            stop_event.set()
            break

    stop_event.set()
    producer.join()

    run_end = time.perf_counter()

    if SHOW_VIEWER:
        cv2.destroyAllWindows()

    produced = buffer.produced
    dropped = max(produced - processed, 0)

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

    elapsed = run_end - run_start
    effective_fps = processed / elapsed

    drop_percent = (
        100.0 * dropped / produced
        if produced > 0
        else 0.0
    )

    result = {
        "model": model_name,
        "method": "tiled_batched",

        "tile_size": TILE_SIZE,
        "overlap": TILE_OVERLAP,
        "nms_iou": TILED_NMS_IOU,
        "confidence": LATENCY_CONF,

        "tiles_per_frame": tiles_per_frame,
        "tile_batch_size": TILE_BATCH_SIZE,
        "forward_passes_per_frame":
            forward_passes_per_frame,

        "source_width": source_width,
        "source_height": source_height,
        "source_fps": source_fps,
        "source_frames": source_frames,
        "frame_period_ms": frame_period_ms,

        "frames_produced": produced,
        "frames_processed": processed,
        "frames_dropped": dropped,
        "drop_percent": drop_percent,

        "effective_fps": effective_fps,

        "predict_mean_ms":
            float(np.mean(predict_times)),

        "predict_median_ms":
            float(np.median(predict_times)),

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
            float(np.mean(e2e_times)),

        "e2e_median_ms":
            float(np.median(e2e_times)),

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
            float(np.max(e2e_times)),

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
        f"  Predict mean: {result['predict_mean_ms']:.2f} ms"
    )
    print(
        f"  E2E mean:     {result['e2e_mean_ms']:.2f} ms"
    )
    print(
        f"  Effective FPS: {result['effective_fps']:.2f}"
    )
    print(
        f"  Drop rate:     {result['drop_percent']:.2f}%"
    )
    print(
        f"  Deadline miss: "
        f"{result['deadline_miss_percent']:.2f}%"
    )

    del model
    del processor

    gc.collect()
    torch.cuda.empty_cache()

    return result


def save_summary(rows):
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
        writer.writerows(rows)


def main():
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available."
        )

    print("=" * 70)
    print(
        "RT-DETRv2 BATCHED TILED "
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

    for model_name, model_id in MODELS.items():
        try:
            row = run_test(
                model_name,
                model_id,
            )

            rows.append(row)
            save_summary(rows)

        except torch.cuda.OutOfMemoryError:
            print()
            print(
                f"OUT OF VRAM: {model_name}"
            )
            print(
                "Change TILE_BATCH_SIZE "
                f"from {TILE_BATCH_SIZE} to 4 "
                "and rerun."
            )

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as error:
            print()
            print(
                f"ERROR: {model_name}"
            )
            print(repr(error))

            gc.collect()
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print(
        "RT-DETRv2 BATCHED TILED "
        "LATENCY COMPLETE"
    )
    print("=" * 70)
    print(
        f"Summary saved to:\n"
        f"{TILED_LATENCY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
