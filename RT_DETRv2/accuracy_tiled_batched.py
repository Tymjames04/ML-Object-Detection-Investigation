import csv
import gc

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

from experiment_config import (
    MODELS,
    DEVICE,
    ACCURACY_CONF,
    ACCURACY_IMAGES_DIR,
    ACCURACY_LABELS_DIR,
    ACCURACY_OUTPUT_DIR,
    TARGET_CLASS_NAMES,
    MODEL_NAME_ALIASES,
    NAME_TO_LOCAL,
    LOCAL_TO_NAME,
    GT_COCO_TO_LOCAL,
    TILE_SIZE,
    TILE_OVERLAP,
    TILED_NMS_IOU,
    TILE_BATCH_SIZE,
    TILED_ACCURACY_SUMMARY_FILE,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def normalize_name(name):
    name = str(name).lower()
    return MODEL_NAME_ALIASES.get(name, name)


def get_image_paths():
    images = sorted(
        p for p in ACCURACY_IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No test images found in {ACCURACY_IMAGES_DIR}")
    return images


def get_label_path(image_path):
    rel = image_path.relative_to(ACCURACY_IMAGES_DIR)
    return (ACCURACY_LABELS_DIR / rel).with_suffix(".txt")


def load_ground_truth(label_path, width, height):
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    boxes, labels = [], []

    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Bad label {label_path}:{line_no}")

            raw_class = int(float(parts[0]))

            # Ignore old boat annotations / any non-target class.
            if raw_class not in GT_COCO_TO_LOCAL:
                continue

            xc, yc, bw, bh = map(float, parts[1:])

            x1 = max(0.0, min((xc - bw / 2) * width, width))
            y1 = max(0.0, min((yc - bh / 2) * height, height))
            x2 = max(0.0, min((xc + bw / 2) * width, width))
            y2 = max(0.0, min((yc + bh / 2) * height, height))

            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid box {label_path}:{line_no}")

            boxes.append([x1, y1, x2, y2])
            labels.append(GT_COCO_TO_LOCAL[raw_class])

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32)
        if boxes else torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64)
        if labels else torch.empty((0,), dtype=torch.int64),
    }


def validate_dataset(image_paths):
    counts = {class_id: 0 for class_id in LOCAL_TO_NAME}
    total = 0

    print("\nValidating accuracy dataset...")

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")

        h, w = image.shape[:2]
        target = load_ground_truth(get_label_path(image_path), w, h)

        for class_id in target["labels"].tolist():
            counts[class_id] += 1
            total += 1

    print(f"  Images:  {len(image_paths)}")
    print(f"  Objects: {total}")
    for class_id in sorted(counts):
        print(f"  {LOCAL_TO_NAME[class_id]:<10} {counts[class_id]}")

    return counts


def get_starts(image_size):
    if image_size <= TILE_SIZE:
        return [0]

    stride = int(TILE_SIZE * (1.0 - TILE_OVERLAP))
    starts = list(range(0, image_size - TILE_SIZE + 1, stride))
    final_start = image_size - TILE_SIZE

    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    return starts


def make_tiles(image_bgr):
    h, w = image_bgr.shape[:2]
    tiles, offsets, sizes = [], [], []

    for y0 in get_starts(h):
        for x0 in get_starts(w):
            tile = image_bgr[
                y0:min(y0 + TILE_SIZE, h),
                x0:min(x0 + TILE_SIZE, w),
            ]
            tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
            th, tw = tile.shape[:2]
            tiles.append(tile)
            offsets.append((x0, y0))
            sizes.append((th, tw))

    return tiles, offsets, sizes


def build_class_map(model):
    mapping = {}

    for label_id, raw_name in model.config.id2label.items():
        name = normalize_name(raw_name)
        if name in TARGET_CLASS_NAMES:
            mapping[int(label_id)] = NAME_TO_LOCAL[name]

    if not mapping:
        raise RuntimeError("No target classes matched model.config.id2label")

    return mapping


def empty_prediction():
    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
    }


def predict_tiled_batched(image_bgr, processor, model, class_map):
    image_h, image_w = image_bgr.shape[:2]
    tiles, offsets, sizes = make_tiles(image_bgr)

    all_boxes, all_scores, all_labels = [], [], []
    forward_passes = 0

    for start in range(0, len(tiles), TILE_BATCH_SIZE):
        end = min(start + TILE_BATCH_SIZE, len(tiles))

        batch_images = tiles[start:end]
        batch_offsets = offsets[start:end]
        batch_sizes = sizes[start:end]

        inputs = processor(
            images=batch_images,
            size={"height": TILE_SIZE, "width": TILE_SIZE},
            return_tensors="pt",
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

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
            threshold=ACCURACY_CONF,
        )

        for prediction, (x0, y0) in zip(predictions, batch_offsets):
            boxes = prediction["boxes"].to(torch.float32)
            scores = prediction["scores"].to(torch.float32)
            model_labels = prediction["labels"].to(torch.int64)

            for model_label, local_label in class_map.items():
                mask = model_labels == model_label
                if not torch.any(mask):
                    continue

                class_boxes = boxes[mask].clone()
                class_scores = scores[mask]

                class_boxes[:, [0, 2]] += x0
                class_boxes[:, [1, 3]] += y0
                class_boxes[:, 0].clamp_(0, image_w)
                class_boxes[:, 2].clamp_(0, image_w)
                class_boxes[:, 1].clamp_(0, image_h)
                class_boxes[:, 3].clamp_(0, image_h)

                valid = (
                    (class_boxes[:, 2] > class_boxes[:, 0])
                    & (class_boxes[:, 3] > class_boxes[:, 1])
                )
                class_boxes = class_boxes[valid]
                class_scores = class_scores[valid]

                if class_boxes.shape[0] == 0:
                    continue

                class_labels = torch.full(
                    (class_boxes.shape[0],),
                    local_label,
                    dtype=torch.int64,
                    device=class_boxes.device,
                )

                all_boxes.append(class_boxes)
                all_scores.append(class_scores)
                all_labels.append(class_labels)

    if not all_boxes:
        return empty_prediction(), len(tiles), forward_passes

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    keep_all = []

    # Class-aware NMS across overlapping tiles.
    for class_id in torch.unique(labels):
        idx = torch.where(labels == class_id)[0]
        keep = nms(boxes[idx], scores[idx], TILED_NMS_IOU)
        keep_all.append(idx[keep])

    if keep_all:
        keep = torch.cat(keep_all)
        keep = keep[torch.argsort(scores[keep], descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    prediction = {
        "boxes": boxes.detach().cpu().to(torch.float32),
        "scores": scores.detach().cpu().to(torch.float32),
        "labels": labels.detach().cpu().to(torch.int64),
    }

    return prediction, len(tiles), forward_passes


def metric_float(metrics, key):
    value = metrics[key]
    if isinstance(value, torch.Tensor):
        value = value.item()
    value = float(value)
    return None if value < 0 else value


def get_per_class_ap(metrics):
    result = {class_id: None for class_id in LOCAL_TO_NAME}

    classes = metrics.get("classes")
    aps = metrics.get("map_per_class")

    if classes is None or aps is None:
        return result

    classes = classes.tolist()
    aps = aps.tolist()

    if not isinstance(classes, list):
        classes = [classes]
    if not isinstance(aps, list):
        aps = [aps]

    for class_id, ap in zip(classes, aps):
        class_id = int(class_id)
        ap = float(ap)
        if class_id in result and ap >= 0:
            result[class_id] = ap

    return result


def run_test(model_name, model_id, image_paths, gt_counts):
    print(f"\nLOADING {model_name}")

    processor = RTDetrImageProcessor.from_pretrained(model_id)
    model = (
        RTDetrV2ForObjectDetection
        .from_pretrained(model_id)
        .to(DEVICE)
        .eval()
    )
    class_map = build_class_map(model)

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} BATCHED TILED {TILE_SIZE}x{TILE_SIZE}")
    print("=" * 70)
    print(f"  Tile size:       {TILE_SIZE}")
    print(f"  Overlap:         {TILE_OVERLAP:.0%}")
    print(f"  Tile batch size: {TILE_BATCH_SIZE}")
    print(f"  Confidence:      {ACCURACY_CONF}")
    print(f"  NMS IoU:         {TILED_NMS_IOU}")

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        backend="faster_coco_eval",
    )
    metric.warn_on_many_detections = False

    processed = 0
    total_tiles = 0
    total_forward_passes = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")

        h, w = image.shape[:2]
        target = load_ground_truth(get_label_path(image_path), w, h)

        prediction, tile_count, forward_passes = predict_tiled_batched(
            image,
            processor,
            model,
            class_map,
        )

        metric.update([prediction], [target])

        processed += 1
        total_tiles += tile_count
        total_forward_passes += forward_passes

        if processed % 25 == 0:
            print(
                f"  Processed {processed}/{len(image_paths)} | "
                f"tiles {total_tiles} | "
                f"forwards {total_forward_passes}"
            )

    results = metric.compute()
    per_class_ap = get_per_class_ap(results)

    row = {
        "model": model_name,
        "method": "tiled_batched",
        "imgsz": TILE_SIZE,
        "tile_size": TILE_SIZE,
        "overlap": TILE_OVERLAP,
        "nms_iou": TILED_NMS_IOU,
        "tile_batch_size": TILE_BATCH_SIZE,
        "confidence": ACCURACY_CONF,
        "images": processed,
        "total_tiles": total_tiles,
        "total_forward_passes": total_forward_passes,
        "avg_tiles_per_image": total_tiles / processed,
        "avg_forward_passes_per_image": total_forward_passes / processed,
        "map50_95": metric_float(results, "map"),
        "map50": metric_float(results, "map_50"),
        "map75": metric_float(results, "map_75"),
        "map_small": metric_float(results, "map_small"),
        "map_medium": metric_float(results, "map_medium"),
        "map_large": metric_float(results, "map_large"),
        "mar100": metric_float(results, "mar_100"),
    }

    for class_id in sorted(LOCAL_TO_NAME):
        name = LOCAL_TO_NAME[class_id]
        row[f"ap_{name}"] = per_class_ap[class_id]
        row[f"gt_{name}"] = gt_counts[class_id]

    print("\n  Results")
    print("  -------")
    print(f"  Images:             {processed}")
    print(f"  Total tiles:        {total_tiles}")
    print(f"  Forward passes:     {total_forward_passes}")
    print(f"  Avg tiles/image:    {row['avg_tiles_per_image']:.2f}")
    print(f"  Avg forwards/image: {row['avg_forward_passes_per_image']:.2f}")
    print(f"  mAP50-95:           {row['map50_95']}")
    print(f"  mAP50:              {row['map50']}")
    print(f"  mAP75:              {row['map75']}")
    print(f"  mAP small:          {row['map_small']}")
    print(f"  mAP medium:         {row['map_medium']}")
    print(f"  mAP large:          {row['map_large']}")
    print(f"  mAR100:             {row['mar100']}")

    for class_id in sorted(LOCAL_TO_NAME):
        print(f"  AP {LOCAL_TO_NAME[class_id]:<9} {per_class_ap[class_id]}")

    del results, metric, model, processor
    gc.collect()
    torch.cuda.empty_cache()

    return row


def save_summary(rows):
    ACCURACY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fields = [
        "model", "method", "imgsz",
        "tile_size", "overlap", "nms_iou", "tile_batch_size", "confidence",
        "images", "total_tiles", "total_forward_passes",
        "avg_tiles_per_image", "avg_forward_passes_per_image",
        "map50_95", "map50", "map75",
        "map_small", "map_medium", "map_large", "mar100",
    ]

    for class_id in sorted(LOCAL_TO_NAME):
        fields.append(f"ap_{LOCAL_TO_NAME[class_id]}")
    for class_id in sorted(LOCAL_TO_NAME):
        fields.append(f"gt_{LOCAL_TO_NAME[class_id]}")

    with open(TILED_ACCURACY_SUMMARY_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is not available.")

    print("=" * 70)
    print("RT-DETRv2 BATCHED TILED FPV ACCURACY BENCHMARK")
    print("=" * 70)
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"Models:          {list(MODELS.keys())}")
    print(f"Tile size:       {TILE_SIZE}")
    print(f"Overlap:         {TILE_OVERLAP:.0%}")
    print(f"Tile batch size: {TILE_BATCH_SIZE}")
    print(f"NMS IoU:         {TILED_NMS_IOU}")
    print(f"Confidence:      {ACCURACY_CONF}")
    print(f"Classes:         {TARGET_CLASS_NAMES}")

    image_paths = get_image_paths()
    gt_counts = validate_dataset(image_paths)
    rows = []

    for model_name, model_id in MODELS.items():
        try:
            row = run_test(
                model_name,
                model_id,
                image_paths,
                gt_counts,
            )
            rows.append(row)
            save_summary(rows)

        except torch.cuda.OutOfMemoryError:
            print(f"\nOUT OF VRAM: {model_name}")
            print(
                f"Change TILE_BATCH_SIZE from {TILE_BATCH_SIZE} to 4 and rerun."
            )
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as error:
            print(f"\nERROR: {model_name}")
            print(repr(error))
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("RT-DETRv2 BATCHED TILED ACCURACY COMPLETE")
    print("=" * 70)
    print(f"Summary saved to:\n{TILED_ACCURACY_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
