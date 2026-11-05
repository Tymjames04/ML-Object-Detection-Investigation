import csv
import gc

import cv2
import numpy as np
import torch

from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from experiment_config import (
    MODELS,
    DEVICE,
    ACCURACY_CONF,
    ACCURACY_IMAGES_DIR,
    ACCURACY_LABELS_DIR,
    ACCURACY_OUTPUT_DIR,
    TILED_ACCURACY_SUMMARY_FILE,
    TARGET_CLASS_NAMES,
    NAME_TO_LOCAL,
    LOCAL_TO_NAME,
    GT_COCO_TO_LOCAL,
    TILE_SIZE,
    TILE_OVERLAP,
    TILED_NMS_IOU,
    TILE_BATCH_SIZE,
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
        for path
        in ACCURACY_IMAGES_DIR.rglob("*")
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

    relative_path = image_path.relative_to(
        ACCURACY_IMAGES_DIR
    )

    return (
        ACCURACY_LABELS_DIR
        / relative_path
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

            # Old boat annotations / any non-target class
            # remain in the label files but are ignored.
            if (
                raw_class_id
                not in GT_COCO_TO_LOCAL
            ):
                continue

            class_id = (
                GT_COCO_TO_LOCAL[
                    raw_class_id
                ]
            )

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

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
    image_bgr,
):

    height, width = (
        image_bgr.shape[:2]
    )

    tiles_rgb = []
    offsets = []

    for y0 in get_starts(
        height
    ):

        for x0 in get_starts(
            width
        ):

            tile_bgr = image_bgr[
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


def empty_prediction():

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


def predict_tiled_batched(
    image_bgr,
    model,
):

    image_height, image_width = (
        image_bgr.shape[:2]
    )

    (
        tiles_rgb,
        offsets,
    ) = make_tiles(
        image_bgr
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

        # RF-DETR officially supports passing a list of images
        # to predict(), which processes the list together in one
        # forward pass and returns one Detections object per image.
        detections_list = model.predict(
            batch_images,
            threshold=ACCURACY_CONF,
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
                    image_width,
                )

                global_box[2].clamp_(
                    0,
                    image_width,
                )

                global_box[1].clamp_(
                    0,
                    image_height,
                )

                global_box[3].clamp_(
                    0,
                    image_height,
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
        return (
            empty_prediction(),
            len(
                tiles_rgb
            ),
            forward_passes,
        )

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

    # Class-aware NMS across overlapping tiles.
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

    prediction = {
        "boxes":
            boxes
            .detach()
            .cpu()
            .to(
                torch.float32
            ),

        "scores":
            scores
            .detach()
            .cpu()
            .to(
                torch.float32
            ),

        "labels":
            labels
            .detach()
            .cpu()
            .to(
                torch.int64
            ),
    }

    return (
        prediction,
        len(
            tiles_rgb
        ),
        forward_passes,
    )


def metric_float(
    metrics,
    key,
):

    value = (
        metrics[
            key
        ]
    )

    if isinstance(
        value,
        torch.Tensor,
    ):
        value = (
            value.item()
        )

    value = float(
        value
    )

    if value < 0:
        return None

    return value


def get_per_class_ap(
    metrics,
):

    per_class = {
        class_id: None
        for class_id
        in LOCAL_TO_NAME
    }

    classes = metrics.get(
        "classes"
    )

    map_per_class = metrics.get(
        "map_per_class"
    )

    if (
        classes is None
        or map_per_class is None
    ):
        return per_class

    classes = classes.tolist()
    map_per_class = (
        map_per_class.tolist()
    )

    if not isinstance(
        classes,
        list,
    ):
        classes = [
            classes
        ]

    if not isinstance(
        map_per_class,
        list,
    ):
        map_per_class = [
            map_per_class
        ]

    for (
        class_id,
        ap,
    ) in zip(
        classes,
        map_per_class,
    ):

        class_id = int(
            class_id
        )

        ap = float(
            ap
        )

        if (
            class_id
            in per_class
            and ap >= 0
        ):
            per_class[
                class_id
            ] = ap

    return per_class


def run_test(
    model_name,
    model_class,
    image_paths,
    ground_truth_counts,
):

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
        f"  Tile size:       "
        f"{TILE_SIZE}"
    )

    print(
        f"  Overlap:         "
        f"{TILE_OVERLAP:.0%}"
    )

    print(
        f"  Tile batch size: "
        f"{TILE_BATCH_SIZE}"
    )

    print(
        f"  Confidence:      "
        f"{ACCURACY_CONF}"
    )

    print(
        f"  NMS IoU:         "
        f"{TILED_NMS_IOU}"
    )

    model = model_class(
        resolution=TILE_SIZE,
        device=DEVICE,
    )

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

        image_bgr = cv2.imread(
            str(image_path)
        )

        if image_bgr is None:
            raise RuntimeError(
                f"Could not read image:\n"
                f"{image_path}"
            )

        (
            image_height,
            image_width,
        ) = image_bgr.shape[:2]

        target = load_ground_truth(
            get_label_path(
                image_path
            ),
            image_width,
            image_height,
        )

        (
            prediction,
            tile_count,
            forward_passes,
        ) = predict_tiled_batched(
            image_bgr,
            model,
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
        total_tiles += tile_count

        total_forward_passes += (
            forward_passes
        )

        if (
            processed % 25
            == 0
        ):
            print(
                f"  Processed "
                f"{processed}/"
                f"{len(image_paths)} "
                f"| tiles "
                f"{total_tiles} "
                f"| forwards "
                f"{total_forward_passes}"
            )

    results = metric.compute()

    per_class_ap = (
        get_per_class_ap(
            results
        )
    )

    row = {
        "model":
            model_name,

        "method":
            "tiled_batched",

        "imgsz":
            TILE_SIZE,

        "tile_size":
            TILE_SIZE,

        "overlap":
            TILE_OVERLAP,

        "nms_iou":
            TILED_NMS_IOU,

        "tile_batch_size":
            TILE_BATCH_SIZE,

        "confidence":
            ACCURACY_CONF,

        "images":
            processed,

        "total_tiles":
            total_tiles,

        "total_forward_passes":
            total_forward_passes,

        "avg_tiles_per_image":
            (
                total_tiles
                / processed
            ),

        "avg_forward_passes_per_image":
            (
                total_forward_passes
                / processed
            ),

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
        f"  Images:             "
        f"{processed}"
    )

    print(
        f"  Total tiles:        "
        f"{total_tiles}"
    )

    print(
        f"  Forward passes:     "
        f"{total_forward_passes}"
    )

    print(
        f"  Avg tiles/image:    "
        f"{row['avg_tiles_per_image']:.2f}"
    )

    print(
        f"  Avg forwards/image: "
        f"{row['avg_forward_passes_per_image']:.2f}"
    )

    print(
        f"  mAP50-95:           "
        f"{row['map50_95']}"
    )

    print(
        f"  mAP50:              "
        f"{row['map50']}"
    )

    print(
        f"  mAP75:              "
        f"{row['map75']}"
    )

    print(
        f"  mAP small:          "
        f"{row['map_small']}"
    )

    print(
        f"  mAP medium:         "
        f"{row['map_medium']}"
    )

    print(
        f"  mAP large:          "
        f"{row['map_large']}"
    )

    print(
        f"  mAR100:             "
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
    del model

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
        "method",
        "imgsz",

        "tile_size",
        "overlap",
        "nms_iou",
        "tile_batch_size",
        "confidence",

        "images",
        "total_tiles",
        "total_forward_passes",
        "avg_tiles_per_image",
        "avg_forward_passes_per_image",

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
        TILED_ACCURACY_SUMMARY_FILE,
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
        "RF-DETR BATCHED TILED "
        "FPV ACCURACY BENCHMARK"
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
        f"{ACCURACY_CONF}"
    )

    print(
        f"Classes:         "
        f"{TARGET_CLASS_NAMES}"
    )

    image_paths = get_image_paths()

    ground_truth_counts = (
        validate_dataset(
            image_paths
        )
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
                image_paths,
                ground_truth_counts,
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
        "ACCURACY COMPLETE"
    )

    print("=" * 70)

    print(
        f"Summary saved to:\n"
        f"{TILED_ACCURACY_SUMMARY_FILE}"
    )


if __name__ == "__main__":
    main()
