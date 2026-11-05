import csv
import gc
from pathlib import Path

import cv2
import torch

from ultralytics import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from experiment_config import (
    MODELS,
    RESOLUTIONS,
    DEVICE,
    RECT,
    QUANTIZE,
    ACCURACY_CONF,
    ACCURACY_IMAGES_DIR,
    ACCURACY_LABELS_DIR,
    ACCURACY_OUTPUT_DIR,
    ACCURACY_SUMMARY_FILE,
    COCO_EVAL_CLASSES,
    COCO_TO_LOCAL,
    LOCAL_TO_NAME,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def get_image_paths():

    if not ACCURACY_IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Image directory does not exist:\n"
            f"{ACCURACY_IMAGES_DIR}"
        )

    image_paths = sorted(
        path
        for path in ACCURACY_IMAGES_DIR.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No test images found in:\n"
            f"{ACCURACY_IMAGES_DIR}"
        )

    return image_paths


def get_label_path(image_path):

    relative = image_path.relative_to(
        ACCURACY_IMAGES_DIR
    )

    return (
        ACCURACY_LABELS_DIR
        / relative
    ).with_suffix(".txt")


def load_ground_truth(
    label_path,
    image_width,
    image_height,
):

    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing ground-truth label:\n"
            f"{label_path}"
        )

    boxes = []
    labels = []

    with open(
        label_path,
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label in "
                    f"{label_path}, line "
                    f"{line_number}:\n"
                    f"{line}"
                )

            raw_class_id = int(
                float(parts[0])
            )

            # Preserve the original label files.
            # Boat (COCO 8) and any other non-target class
            # are ignored for this 3-class experiment.
            if (
                raw_class_id
                not in COCO_TO_LOCAL
            ):
                continue

            class_id = (
                COCO_TO_LOCAL[
                    raw_class_id
                ]
            )

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            for value in (
                x_center,
                y_center,
                box_width,
                box_height,
            ):
                if (
                    value < 0.0
                    or value > 1.0
                ):
                    raise ValueError(
                        f"Normalized coordinate outside "
                        f"0-1 in {label_path}, "
                        f"line {line_number}"
                    )

            x1 = (
                x_center
                - box_width / 2.0
            ) * image_width

            y1 = (
                y_center
                - box_height / 2.0
            ) * image_height

            x2 = (
                x_center
                + box_width / 2.0
            ) * image_width

            y2 = (
                y_center
                + box_height / 2.0
            ) * image_height

            x1 = max(
                0.0,
                min(x1, image_width),
            )

            y1 = max(
                0.0,
                min(y1, image_height),
            )

            x2 = max(
                0.0,
                min(x2, image_width),
            )

            y2 = max(
                0.0,
                min(y2, image_height),
            )

            if (
                x2 <= x1
                or y2 <= y1
            ):
                raise ValueError(
                    f"Invalid zero/negative-area "
                    f"box in {label_path}, "
                    f"line {line_number}"
                )

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ]
            )

            labels.append(
                class_id
            )

    if boxes:
        return {
            "boxes":
                torch.tensor(
                    boxes,
                    dtype=torch.float32,
                ),

            "labels":
                torch.tensor(
                    labels,
                    dtype=torch.int64,
                ),
        }

    return {
        "boxes":
            torch.empty(
                (0, 4),
                dtype=torch.float32,
            ),

        "labels":
            torch.empty(
                (0,),
                dtype=torch.int64,
            ),
    }


def validate_dataset(
    image_paths,
):

    print()
    print(
        "Validating accuracy dataset..."
    )

    class_counts = {
        class_id: 0
        for class_id
        in LOCAL_TO_NAME
    }

    total_objects = 0
    empty_images = 0

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        target = load_ground_truth(
            get_label_path(
                image_path
            ),
            width,
            height,
        )

        if (
            len(
                target[
                    "labels"
                ]
            )
            == 0
        ):
            empty_images += 1

        for class_id in (
            target[
                "labels"
            ].tolist()
        ):

            class_counts[
                class_id
            ] += 1

            total_objects += 1

    print(
        f"  Images:       "
        f"{len(image_paths)}"
    )

    print(
        f"  Objects:      "
        f"{total_objects}"
    )

    print(
        f"  Empty images: "
        f"{empty_images}"
    )

    print()

    for class_id in sorted(
        class_counts
    ):

        print(
            f"  "
            f"{LOCAL_TO_NAME[class_id]:<10} "
            f"{class_counts[class_id]}"
        )

    return class_counts


def convert_yolo_predictions(
    result,
):

    boxes = result.boxes

    if (
        boxes is None
        or len(boxes) == 0
    ):

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
        }

    xyxy = (
        boxes.xyxy
        .detach()
        .cpu()
        .to(
            torch.float32
        )
    )

    scores = (
        boxes.conf
        .detach()
        .cpu()
        .to(
            torch.float32
        )
    )

    coco_classes = (
        boxes.cls
        .detach()
        .cpu()
        .to(
            torch.int64
        )
    )

    keep_indices = []
    local_labels = []

    for index, class_id in enumerate(
        coco_classes.tolist()
    ):

        class_id = int(
            class_id
        )

        if (
            class_id
            not in COCO_TO_LOCAL
        ):
            continue

        keep_indices.append(
            index
        )

        local_labels.append(
            COCO_TO_LOCAL[
                class_id
            ]
        )

    if not keep_indices:

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
        }

    keep_indices = torch.tensor(
        keep_indices,
        dtype=torch.long,
    )

    return {
        "boxes":
            xyxy[
                keep_indices
            ],

        "scores":
            scores[
                keep_indices
            ],

        "labels":
            torch.tensor(
                local_labels,
                dtype=torch.int64,
            ),
    }


def metric_float(
    metrics,
    key,
):

    value = metrics[
        key
    ]

    if isinstance(
        value,
        torch.Tensor,
    ):
        value = value.item()

    value = float(
        value
    )

    return (
        None
        if value < 0
        else value
    )


def get_per_class_ap(
    metrics,
):

    result = {
        class_id: None
        for class_id
        in LOCAL_TO_NAME
    }

    classes = metrics.get(
        "classes"
    )

    aps = metrics.get(
        "map_per_class"
    )

    if (
        classes is None
        or aps is None
    ):
        return result

    classes = classes.tolist()
    aps = aps.tolist()

    if not isinstance(
        classes,
        list,
    ):
        classes = [
            classes
        ]

    if not isinstance(
        aps,
        list,
    ):
        aps = [
            aps
        ]

    for class_id, ap in zip(
        classes,
        aps,
    ):

        class_id = int(
            class_id
        )

        ap = float(
            ap
        )

        if (
            class_id
            in result
            and ap >= 0
        ):
            result[
                class_id
            ] = ap

    return result


def run_trial(
    model,
    model_name,
    imgsz,
    image_paths,
    ground_truth_counts,
):

    print()
    print("=" * 70)

    print(
        f"{model_name.upper()} "
        f"@ {imgsz}"
    )

    print("=" * 70)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        backend="faster_coco_eval",
    )

    metric.warn_on_many_detections = False

    processed = 0

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        height, width = (
            image.shape[:2]
        )

        target = load_ground_truth(
            get_label_path(
                image_path
            ),
            width,
            height,
        )

        result = model.predict(
            source=image,
            imgsz=imgsz,
            conf=ACCURACY_CONF,
            classes=COCO_EVAL_CLASSES,
            device=DEVICE,
            rect=RECT,
            quantize=QUANTIZE,
            verbose=False,
        )[0]

        prediction = (
            convert_yolo_predictions(
                result
            )
        )

        metric.update(
            [
                prediction
            ],
            [
                target
            ],
        )

        processed += 1

        if (
            processed % 100
            == 0
        ):
            print(
                f"  Processed "
                f"{processed}/"
                f"{len(image_paths)}"
            )

    results = metric.compute()

    per_class_ap = (
        get_per_class_ap(
            results
        )
    )

    row = {
        "model":
            Path(
                model_name
            ).stem,

        "imgsz":
            imgsz,

        "images":
            processed,

        "map50_95":
            metric_float(
                results,
                "map",
            ),

        "map50":
            metric_float(
                results,
                "map_50",
            ),

        "map75":
            metric_float(
                results,
                "map_75",
            ),

        "map_small":
            metric_float(
                results,
                "map_small",
            ),

        "map_medium":
            metric_float(
                results,
                "map_medium",
            ),

        "map_large":
            metric_float(
                results,
                "map_large",
            ),

        "mar100":
            metric_float(
                results,
                "mar_100",
            ),
    }

    for class_id in sorted(
        LOCAL_TO_NAME
    ):

        name = (
            LOCAL_TO_NAME[
                class_id
            ]
        )

        row[
            f"ap_{name}"
        ] = (
            per_class_ap[
                class_id
            ]
        )

        row[
            f"gt_{name}"
        ] = (
            ground_truth_counts[
                class_id
            ]
        )

    print()
    print("  Results")
    print("  -------")

    print(
        f"  Images:       "
        f"{processed}"
    )

    print(
        f"  mAP50-95:     "
        f"{row['map50_95']}"
    )

    print(
        f"  mAP50:        "
        f"{row['map50']}"
    )

    print(
        f"  mAP75:        "
        f"{row['map75']}"
    )

    print(
        f"  mAP small:    "
        f"{row['map_small']}"
    )

    print(
        f"  mAP medium:   "
        f"{row['map_medium']}"
    )

    print(
        f"  mAP large:    "
        f"{row['map_large']}"
    )

    print(
        f"  mAR100:       "
        f"{row['mar100']}"
    )

    print()

    for class_id in sorted(
        LOCAL_TO_NAME
    ):

        print(
            f"  AP "
            f"{LOCAL_TO_NAME[class_id]:<9} "
            f"{per_class_ap[class_id]}"
        )

    del results
    del metric

    gc.collect()
    torch.cuda.empty_cache()

    return row


def save_summary(
    rows,
):

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

    for class_id in sorted(
        LOCAL_TO_NAME
    ):
        fieldnames.append(
            f"ap_"
            f"{LOCAL_TO_NAME[class_id]}"
        )

    for class_id in sorted(
        LOCAL_TO_NAME
    ):
        fieldnames.append(
            f"gt_"
            f"{LOCAL_TO_NAME[class_id]}"
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

        writer.writerows(
            rows
        )


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available."
        )

    print("=" * 70)

    print(
        "YOLO26 FPV ACCURACY "
        "BENCHMARK - 3 CLASS"
    )

    print("=" * 70)

    print(
        f"GPU:        "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"PyTorch:    "
        f"{torch.__version__}"
    )

    print(
        f"ROCm/HIP:   "
        f"{torch.version.hip}"
    )

    print(
        f"Models:     "
        f"{MODELS}"
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
        f"{COCO_EVAL_CLASSES}"
    )

    image_paths = get_image_paths()

    ground_truth_counts = (
        validate_dataset(
            image_paths
        )
    )

    summary_rows = []

    total_runs = (
        len(
            MODELS
        )
        * len(
            RESOLUTIONS
        )
    )

    run_number = 0

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

            run_number += 1

            print()
            print(
                f"RUN "
                f"{run_number}/"
                f"{total_runs}"
            )

            try:

                row = run_trial(
                    model,
                    model_name,
                    imgsz,
                    image_paths,
                    ground_truth_counts,
                )

                summary_rows.append(
                    row
                )

                save_summary(
                    summary_rows
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
        "YOLO26 ACCURACY "
        "BENCHMARK COMPLETE"
    )

    print("=" * 70)

    print(
        f"Summary saved to:\n"
        f"{ACCURACY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
