import csv
import gc

import cv2
import torch

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoImageProcessor, DFineForObjectDetection

from experiment_config import (
    MODELS,
    RESOLUTIONS,
    DEVICE,
    ACCURACY_CONF,
    ACCURACY_IMAGES_DIR,
    ACCURACY_LABELS_DIR,
    ACCURACY_OUTPUT_DIR,
    ACCURACY_SUMMARY_FILE,
    TARGET_CLASS_NAMES,
    MODEL_NAME_ALIASES,
    NAME_TO_LOCAL,
    LOCAL_TO_NAME,
    GT_COCO_TO_LOCAL,
)




IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_image_paths():
    if not ACCURACY_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n{ACCURACY_IMAGES_DIR}"
        )

    image_paths = sorted(
        p
        for p in ACCURACY_IMAGES_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"No test images found in:\n{ACCURACY_IMAGES_DIR}"
        )

    return image_paths


def get_label_path(image_path):
    relative = image_path.relative_to(ACCURACY_IMAGES_DIR)
    return (ACCURACY_LABELS_DIR / relative).with_suffix(".txt")


def load_ground_truth(label_path, image_width, image_height):
    """
    Ground truth stays in the original YOLO files with COCO IDs.
    Any class absent from GT_COCO_TO_LOCAL is ignored.

    For the 3-class experiment:
        COCO 0 -> local 0 person
        COCO 2 -> local 1 car
        COCO 4 -> local 2 airplane

    Old boat annotations (COCO 8) may remain in the files and are ignored.
    """
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing ground-truth label:\n{label_path}"
        )

    boxes = []
    labels = []

    with open(label_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label in {label_path}, "
                    f"line {line_number}:\n{line}"
                )

            raw_class_id = int(float(parts[0]))

            # Ignore classes not used by the current experiment.
            if raw_class_id not in GT_COCO_TO_LOCAL:
                continue

            local_class_id = GT_COCO_TO_LOCAL[raw_class_id]

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            values = (xc, yc, bw, bh)
            if any(v < 0.0 or v > 1.0 for v in values):
                raise ValueError(
                    f"Normalized coordinate outside 0-1 in "
                    f"{label_path}, line {line_number}"
                )

            x1 = (xc - bw / 2.0) * image_width
            y1 = (yc - bh / 2.0) * image_height
            x2 = (xc + bw / 2.0) * image_width
            y2 = (yc + bh / 2.0) * image_height

            x1 = max(0.0, min(x1, image_width))
            y1 = max(0.0, min(y1, image_height))
            x2 = max(0.0, min(x2, image_width))
            y2 = max(0.0, min(y2, image_height))

            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Invalid zero/negative-area box in "
                    f"{label_path}, line {line_number}"
                )

            boxes.append([x1, y1, x2, y2])
            labels.append(local_class_id)

    if boxes:
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
    }


def normalize_class_name(class_name):
    class_name = str(class_name).lower()
    return MODEL_NAME_ALIASES.get(class_name, class_name)


def convert_predictions(prediction, model):
    boxes = prediction["boxes"].detach().cpu().to(torch.float32)
    scores = prediction["scores"].detach().cpu().to(torch.float32)
    model_labels = prediction["labels"].detach().cpu().to(torch.int64)

    kept_boxes = []
    kept_scores = []
    kept_labels = []

    for box, score, model_label in zip(boxes, scores, model_labels):
        label_id = int(model_label.item())
        raw_name = model.config.id2label[label_id]
        class_name = normalize_class_name(raw_name)

        if class_name not in TARGET_CLASS_NAMES:
            continue

        kept_boxes.append(box)
        kept_scores.append(score)
        kept_labels.append(NAME_TO_LOCAL[class_name])

    if kept_boxes:
        return {
            "boxes": torch.stack(kept_boxes),
            "scores": torch.stack(kept_scores),
            "labels": torch.tensor(kept_labels, dtype=torch.int64),
        }

    return {
        "boxes": torch.empty((0, 4), dtype=torch.float32),
        "scores": torch.empty((0,), dtype=torch.float32),
        "labels": torch.empty((0,), dtype=torch.int64),
    }


def validate_dataset(image_paths):
    print()
    print("Validating accuracy dataset...")

    class_counts = {
        class_id: 0
        for class_id in LOCAL_TO_NAME
    }

    total_objects = 0
    empty_images = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(
                f"Could not read image:\n{image_path}"
            )

        h, w = image.shape[:2]

        target = load_ground_truth(
            get_label_path(image_path),
            w,
            h,
        )

        if len(target["labels"]) == 0:
            empty_images += 1

        for class_id in target["labels"].tolist():
            class_counts[class_id] += 1
            total_objects += 1

    print(f"  Images:       {len(image_paths)}")
    print(f"  Objects:      {total_objects}")
    print(f"  Empty images: {empty_images}")
    print()

    for class_id in sorted(class_counts):
        print(
            f"  {LOCAL_TO_NAME[class_id]:<10} "
            f"{class_counts[class_id]}"
        )

    return class_counts


def metric_float(metrics, key):
    value = metrics[key]
    if isinstance(value, torch.Tensor):
        value = value.item()

    value = float(value)
    return None if value < 0 else value


def get_per_class_ap(metrics):
    per_class = {
        class_id: None
        for class_id in LOCAL_TO_NAME
    }

    classes = metrics.get("classes")
    map_per_class = metrics.get("map_per_class")

    if classes is None or map_per_class is None:
        return per_class

    classes = classes.tolist()
    map_per_class = map_per_class.tolist()

    if not isinstance(classes, list):
        classes = [classes]

    if not isinstance(map_per_class, list):
        map_per_class = [map_per_class]

    for class_id, ap in zip(classes, map_per_class):
        class_id = int(class_id)
        ap = float(ap)

        if class_id in per_class and ap >= 0:
            per_class[class_id] = ap

    return per_class


def run_accuracy_test(
    model_name,
    model_id,
    resolution,
    image_paths,
    ground_truth_counts,
):
    print()
    print("=" * 70)
    print(f"{model_name.upper()} @ {resolution}x{resolution}")
    print("=" * 70)

    processor = AutoImageProcessor.from_pretrained(model_id)

    model = (
        DFineForObjectDetection
        .from_pretrained(model_id)
        .to(DEVICE)
        .eval()
    )

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        backend="faster_coco_eval",
    )
    metric.warn_on_many_detections = False

    processed = 0

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(
                f"Could not read image:\n{image_path}"
            )

        h, w = image_bgr.shape[:2]

        image_rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )

        target = load_ground_truth(
            get_label_path(image_path),
            w,
            h,
        )

        inputs = processor(
            images=image_rgb,
            size={
                "height": resolution,
                "width": resolution,
            },
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        target_sizes = torch.tensor(
            [[h, w]],
            device=DEVICE,
        )

        prediction = (
            processor
            .post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=ACCURACY_CONF,
            )[0]
        )

        prediction = convert_predictions(
            prediction,
            model,
        )

        metric.update(
            [prediction],
            [target],
        )

        processed += 1

        if processed % 100 == 0:
            print(
                f"  Processed "
                f"{processed}/{len(image_paths)}"
            )

    results = metric.compute()
    per_class_ap = get_per_class_ap(results)

    row = {
        "model": model_name,
        "imgsz": resolution,
        "images": processed,
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
        row[f"gt_{name}"] = ground_truth_counts[class_id]

    print()
    print(f"  Images:       {processed}")
    print(f"  mAP50-95:     {row['map50_95']}")
    print(f"  mAP50:        {row['map50']}")
    print(f"  mAP75:        {row['map75']}")
    print(f"  mAP small:    {row['map_small']}")
    print(f"  mAP medium:   {row['map_medium']}")
    print(f"  mAP large:    {row['map_large']}")
    print(f"  mAR100:       {row['mar100']}")
    print()

    for class_id in sorted(LOCAL_TO_NAME):
        name = LOCAL_TO_NAME[class_id]
        print(
            f"  AP {name:<9} "
            f"{per_class_ap[class_id]}"
        )

    del results
    del metric
    del model
    del processor

    gc.collect()
    torch.cuda.empty_cache()

    return row


def save_summary(rows):
    ACCURACY_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "model",
        "imgsz",
        "images",
        "map50_95",
        "map50",
        "map75",
        "map_small",
        "map_medium",
        "map_large",
        "mar100",
    ]

    for class_id in sorted(LOCAL_TO_NAME):
        fieldnames.append(
            f"ap_{LOCAL_TO_NAME[class_id]}"
        )

    for class_id in sorted(LOCAL_TO_NAME):
        fieldnames.append(
            f"gt_{LOCAL_TO_NAME[class_id]}"
        )

    with open(
        ACCURACY_SUMMARY_FILE,
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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available."
        )

    print("=" * 70)
    print("D-FINE FPV ACCURACY BENCHMARK - 3 CLASS")
    print("=" * 70)

    print(
        f"GPU:        "
        f"{torch.cuda.get_device_name(0)}"
    )
    print(
        f"Models:     "
        f"{list(MODELS.keys())}"
    )
    print(
        f"Sizes:      "
        f"{RESOLUTIONS}"
    )
    print(
        f"Confidence: "
        f"{ACCURACY_CONF}"
    )
    print(
        f"Classes:    "
        f"{TARGET_CLASS_NAMES}"
    )

    image_paths = get_image_paths()
    ground_truth_counts = validate_dataset(
        image_paths
    )

    summary_rows = []

    total_runs = (
        len(MODELS)
        * len(RESOLUTIONS)
    )

    run_number = 0

    for model_name, model_id in MODELS.items():
        for resolution in RESOLUTIONS:
            run_number += 1

            print()
            print(
                f"RUN "
                f"{run_number}/{total_runs}"
            )

            try:
                row = run_accuracy_test(
                    model_name,
                    model_id,
                    resolution,
                    image_paths,
                    ground_truth_counts,
                )

                summary_rows.append(row)
                save_summary(summary_rows)

            except Exception as error:
                print()
                print(
                    f"ERROR: "
                    f"{model_name} "
                    f"@ {resolution}"
                )
                print(repr(error))

                gc.collect()
                torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("ACCURACY BENCHMARK COMPLETE")
    print("=" * 70)
    print(
        f"Summary saved to:\n"
        f"{ACCURACY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
