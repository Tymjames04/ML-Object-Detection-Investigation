from pathlib import Path
import threading
import time

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from rfdetr import (
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRLarge,
)


# =========================
# CONFIG - EDIT THESE
# =========================

ROOT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = ROOT_DIR / "videos" / "30fps.mp4"

MODEL_NAME = "rfdetr_medium"

MODELS = {
    "rfdetr_nano": RFDETRNano,
    "rfdetr_small": RFDETRSmall,
    "rfdetr_medium": RFDETRMedium,
    "rfdetr_large": RFDETRLarge,
}

# Transformer tile/input resolution.
TILE_SIZE = 640


# =========================
# DEPLOYMENT SETTINGS
# =========================

DEVICE = "cuda"

CONFIDENCE = 0.3

TILE_OVERLAP = 0.20
TILE_BATCH_SIZE = 8
NMS_IOU = 0.50

WARMUP_RUNS = 3

DISPLAY_SCALE = 1.0

# Leave False for normal use.
DRAW_TILE_GRID = False

WINDOW_NAME = "RF-DETR Batched Tiled Viewer"

TARGET_CLASS_NAMES = (
    "person",
    "car",
    "airplane",
)

MODEL_NAME_ALIASES = {
    "aeroplane": "airplane",
}

NAME_TO_LOCAL = {
    "person": 0,
    "car": 1,
    "airplane": 2,
}

LOCAL_TO_NAME = {
    0: "person",
    1: "car",
    2: "airplane",
}


class LatestFrameBuffer:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = -1
        self.produced = 0
        self.finished = False

    def publish(
        self,
        frame_id,
        frame,
    ):
        with self.condition:
            self.frame_id = frame_id
            self.frame = frame
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
                )

            return None


def video_producer(
    video_path,
    source_fps,
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
            time.sleep(
                delay
            )

        if stop_event.is_set():
            break

        buffer.publish(
            frame_id,
            frame,
        )

        frame_id += 1

    cap.release()
    buffer.finish()


def normalize_name(name):
    name = str(
        name
    ).lower()

    return MODEL_NAME_ALIASES.get(
        name,
        name,
    )


def get_starts(
    image_size,
):
    if image_size <= TILE_SIZE:
        return [0]

    stride = int(
        TILE_SIZE
        * (1.0 - TILE_OVERLAP)
    )

    if stride <= 0:
        raise ValueError(
            "TILE_OVERLAP must be less than 1.0"
        )

    starts = list(
        range(
            0,
            image_size - TILE_SIZE + 1,
            stride,
        )
    )

    final_start = (
        image_size - TILE_SIZE
    )

    if (
        not starts
        or starts[-1] != final_start
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
                y0:y0 + TILE_SIZE,
                x0:x0 + TILE_SIZE,
            ]

            tile_rgb = cv2.cvtColor(
                tile_bgr,
                cv2.COLOR_BGR2RGB,
            )

            tiles_rgb.append(
                tile_rgb
            )

            offsets.append(
                (x0, y0)
            )

    return (
        tiles_rgb,
        offsets,
    )


def empty_prediction(
    tile_count=0,
    forward_passes=0,
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

        "tiles":
            tile_count,

        "forward_passes":
            forward_passes,
    }


def batched_tiled_predict(
    frame_bgr,
    model,
):
    frame_h, frame_w = (
        frame_bgr.shape[:2]
    )

    tiles, offsets = (
        make_tiles(
            frame_bgr
        )
    )

    all_boxes = []
    all_scores = []
    all_labels = []

    forward_passes = 0

    for start in range(
        0,
        len(tiles),
        TILE_BATCH_SIZE,
    ):
        end = min(
            start + TILE_BATCH_SIZE,
            len(tiles),
        )

        batch_images = (
            tiles[start:end]
        )

        batch_offsets = (
            offsets[start:end]
        )

        detections_list = (
            model.predict(
                batch_images,
                threshold=CONFIDENCE,
                include_source_image=False,
            )
        )

        forward_passes += 1

        if not isinstance(
            detections_list,
            (list, tuple),
        ):
            if len(batch_images) != 1:
                raise RuntimeError(
                    "RF-DETR batch prediction "
                    "did not return one result "
                    "per tile."
                )

            detections_list = [
                detections_list
            ]

        if (
            len(detections_list)
            != len(batch_images)
        ):
            raise RuntimeError(
                "RF-DETR returned a different "
                "number of results than input tiles."
            )

        for (
            detections,
            (x0, y0),
        ) in zip(
            detections_list,
            batch_offsets,
        ):
            if len(detections) == 0:
                continue

            class_names = (
                detections.data.get(
                    "class_name"
                )
            )

            if class_names is None:
                raise RuntimeError(
                    "RF-DETR detections did not "
                    "contain class_name."
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
                index,
                raw_name,
            ) in enumerate(
                class_names
            ):
                name = normalize_name(
                    raw_name
                )

                if (
                    name
                    not in TARGET_CLASS_NAMES
                ):
                    continue

                box = torch.tensor(
                    boxes_np[
                        index
                    ],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                score = torch.tensor(
                    scores_np[
                        index
                    ],
                    dtype=torch.float32,
                    device=DEVICE,
                )

                box[0] += x0
                box[2] += x0
                box[1] += y0
                box[3] += y0

                box[0].clamp_(
                    0,
                    frame_w,
                )
                box[2].clamp_(
                    0,
                    frame_w,
                )
                box[1].clamp_(
                    0,
                    frame_h,
                )
                box[3].clamp_(
                    0,
                    frame_h,
                )

                if (
                    box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    continue

                all_boxes.append(
                    box
                )

                all_scores.append(
                    score
                )

                all_labels.append(
                    NAME_TO_LOCAL[
                        name
                    ]
                )

    if not all_boxes:
        return empty_prediction(
            tile_count=len(
                tiles
            ),
            forward_passes=(
                forward_passes
            ),
        )

    boxes = torch.stack(
        all_boxes
    )

    scores = torch.stack(
        all_scores
    )

    labels = torch.tensor(
        all_labels,
        dtype=torch.int64,
        device=DEVICE,
    )

    keep_all = []

    for class_id in torch.unique(
        labels
    ):
        indices = torch.where(
            labels == class_id
        )[0]

        keep = nms(
            boxes[
                indices
            ],
            scores[
                indices
            ],
            NMS_IOU,
        )

        keep_all.append(
            indices[
                keep
            ]
        )

    keep = torch.cat(
        keep_all
    )

    keep = keep[
        torch.argsort(
            scores[
                keep
            ],
            descending=True,
        )
    ]

    return {
        "boxes":
            boxes[
                keep
            ]
            .detach()
            .cpu()
            .to(
                torch.float32
            ),

        "scores":
            scores[
                keep
            ]
            .detach()
            .cpu()
            .to(
                torch.float32
            ),

        "labels":
            labels[
                keep
            ]
            .detach()
            .cpu()
            .to(
                torch.int64
            ),

        "tiles":
            len(
                tiles
            ),

        "forward_passes":
            forward_passes,
    }


def draw_predictions(
    frame_bgr,
    prediction,
    status_text,
):
    frame = frame_bgr.copy()

    for (
        box,
        score,
        label,
    ) in zip(
        prediction[
            "boxes"
        ].numpy(),
        prediction[
            "scores"
        ].numpy(),
        prediction[
            "labels"
        ].numpy(),
    ):
        x1, y1, x2, y2 = (
            box.astype(
                int
            )
        )

        name = LOCAL_TO_NAME[
            int(
                label
            )
        ]

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
            (
                x1,
                max(
                    18,
                    y1 - 6,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if DRAW_TILE_GRID:
        height, width = (
            frame.shape[:2]
        )

        for y0 in get_starts(
            height
        ):
            for x0 in get_starts(
                width
            ):
                cv2.rectangle(
                    frame,
                    (x0, y0),
                    (
                        min(
                            x0 + TILE_SIZE,
                            width,
                        ) - 1,
                        min(
                            y0 + TILE_SIZE,
                            height,
                        ) - 1,
                    ),
                    (255, 255, 0),
                    1,
                )

    cv2.putText(
        frame,
        status_text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def show_frame(frame):
    if DISPLAY_SCALE != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )

    cv2.imshow(
        WINDOW_NAME,
        frame,
    )


def warmup(
    frame,
    model,
):
    print(
        f"Warming up "
        f"{WARMUP_RUNS} runs..."
    )

    for _ in range(
        WARMUP_RUNS
    ):
        _ = batched_tiled_predict(
            frame,
            model,
        )

    torch.cuda.synchronize()


def main():
    if MODEL_NAME not in MODELS:
        raise ValueError(
            f"Unknown MODEL_NAME: "
            f"{MODEL_NAME}\n"
            f"Choose one of: "
            f"{list(MODELS)}"
        )

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            VIDEO_PATH
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available."
        )

    cap = cv2.VideoCapture(
        str(
            VIDEO_PATH
        )
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: "
            f"{VIDEO_PATH}"
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

    ret, warmup_frame = (
        cap.read()
    )

    cap.release()

    if not ret:
        raise RuntimeError(
            "Could not read a warmup frame."
        )

    if source_fps <= 0:
        raise RuntimeError(
            "Invalid source FPS."
        )

    model_class = (
        MODELS[
            MODEL_NAME
        ]
    )

    print("=" * 70)
    print(
        "RF-DETR BATCHED TILED "
        "STANDALONE VIEWER"
    )
    print("=" * 70)

    print(
        f"Video:       "
        f"{VIDEO_PATH}"
    )

    print(
        f"Source:      "
        f"{source_width}x"
        f"{source_height} "
        f"@ {source_fps:.3f} FPS"
    )

    print(
        f"Model:       "
        f"{MODEL_NAME}"
    )

    print(
        f"Tile size:   "
        f"{TILE_SIZE}"
    )

    print(
        f"Overlap:     "
        f"{TILE_OVERLAP:.0%}"
    )

    print(
        f"Tile batch:  "
        f"{TILE_BATCH_SIZE}"
    )

    print(
        f"Confidence:  "
        f"{CONFIDENCE}"
    )

    print(
        f"NMS IoU:     "
        f"{NMS_IOU}"
    )

    print(
        "Controls:    "
        "q or Esc = quit"
    )

    model = model_class(
        resolution=TILE_SIZE,
        device=DEVICE,
    )

    warmup(
        warmup_frame,
        model,
    )

    buffer = LatestFrameBuffer()
    stop_event = threading.Event()

    producer_thread = (
        threading.Thread(
            target=video_producer,
            args=(
                VIDEO_PATH,
                source_fps,
                buffer,
                stop_event,
            ),
            daemon=True,
        )
    )

    producer_thread.start()

    last_frame_id = -1
    processed = 0

    viewer_start = (
        time.perf_counter()
    )

    while True:
        item = buffer.get_latest(
            last_frame_id
        )

        if item is None:
            break

        frame_id, frame = item

        last_frame_id = (
            frame_id
        )

        torch.cuda.synchronize()

        start = (
            time.perf_counter()
        )

        prediction = (
            batched_tiled_predict(
                frame,
                model,
            )
        )

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

        status = (
            f"{MODEL_NAME} | "
            f"{prediction['tiles']} tiles | "
            f"{prediction['forward_passes']} forward | "
            f"{predict_ms:.1f} ms | "
            f"{processed_fps:.1f} processed FPS"
        )

        annotated = (
            draw_predictions(
                frame,
                prediction,
                status,
            )
        )

        show_frame(
            annotated
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
