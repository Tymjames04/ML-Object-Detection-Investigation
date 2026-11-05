from pathlib import Path
import csv
import threading
import time
import os
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


# ============================================================
# CONFIG - EDIT THESE
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

# Camera / virtual-camera index. May differ between systems.
CAPTURE_DEVICE = int(os.getenv("CAPTURE_DEVICE", "0"))

CAPTURE_BACKEND = (
    cv2.CAP_DSHOW
    if os.name == "nt"
    else cv2.CAP_ANY
)

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
CAPTURE_FPS = 30.0

RUN_SECONDS = None

MODEL_NAME = "rfdetr_medium"

MODELS = {
    "rfdetr_nano": RFDETRNano,
    "rfdetr_small": RFDETRSmall,
    "rfdetr_medium": RFDETRMedium,
    "rfdetr_large": RFDETRLarge,
}

TILE_SIZE = 640
TILE_OVERLAP = 0.20
TILE_BATCH_SIZE = 8
NMS_IOU = 0.50

CONFIDENCE = 0.3

WARMUP_RUNS = 20
DEVICE = "cuda"

MAX_DETECTION_AGE_MS = 200.0

SHOW_DIAGNOSTICS = True
DISPLAY_SCALE = 1.0

WINDOW_NAME = "RF-DETR Smooth OBS Demo"

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "obs_smooth_demo"
)

DISPLAY_MODEL_NAME = MODEL_NAME

TARGET_CLASS_NAMES = (
    "person",
    "car",
    "airplane",
)

MODEL_NAME_ALIASES = {
    "person": "person",
    "car": "car",
    "airplane": "airplane",
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



class CaptureState:
    def __init__(self):
        self.condition = threading.Condition()
        self.frame_id = -1
        self.frame = None
        self.arrival_time = None
        self.frames_captured = 0
        self.finished = False
        self.first_capture_time = None
        self.last_capture_time = None

    def publish(self, frame):
        now = time.perf_counter()

        with self.condition:
            self.frame_id += 1
            self.frame = frame
            self.arrival_time = now
            self.frames_captured += 1

            if self.first_capture_time is None:
                self.first_capture_time = now

            self.last_capture_time = now
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            self.finished = True
            self.condition.notify_all()

    def wait_for_newer(self, last_frame_id):
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

    def snapshot(self):
        with self.condition:
            if self.frame_id < 0:
                return None

            return (
                self.frame_id,
                self.frame,
                self.arrival_time,
            )


class PredictionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.prediction = None
        self.source_frame_id = -1
        self.source_arrival_time = None
        self.completed_time = None

    def publish(
        self,
        prediction,
        source_frame_id,
        source_arrival_time,
    ):
        with self.lock:
            self.prediction = prediction
            self.source_frame_id = source_frame_id
            self.source_arrival_time = source_arrival_time
            self.completed_time = time.perf_counter()

    def snapshot(self):
        with self.lock:
            if self.prediction is None:
                return None

            return (
                self.prediction,
                self.source_frame_id,
                self.source_arrival_time,
                self.completed_time,
            )


class InferenceStats:
    def __init__(self):
        self.predict_ms = []
        self.e2e_ms = []
        self.detection_counts = []
        self.processed = 0


def open_capture():
    cap = cv2.VideoCapture(
        CAPTURE_DEVICE,
        CAPTURE_BACKEND,
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open capture device {CAPTURE_DEVICE}. "
            f"Try a different CAPTURE_DEVICE index."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAPTURE_WIDTH,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAPTURE_HEIGHT,
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        CAPTURE_FPS,
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    return cap


def capture_worker(
    cap,
    capture_state,
    stop_event,
):
    while not stop_event.is_set():
        ok, frame = cap.read()

        if not ok:
            break

        capture_state.publish(
            frame
        )

    capture_state.finish()


def actual_capture_fps(
    capture_state,
):
    if (
        capture_state.frames_captured < 2
        or capture_state.first_capture_time is None
        or capture_state.last_capture_time is None
    ):
        return 0.0

    elapsed = (
        capture_state.last_capture_time
        - capture_state.first_capture_time
    )

    if elapsed <= 0:
        return 0.0

    return (
        capture_state.frames_captured - 1
    ) / elapsed


def save_csv(
    output_path,
    row,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                row.keys()
            ),
        )

        writer.writeheader()
        writer.writerow(
            row
        )


def metric_mean(values):
    if not values:
        return None

    return float(
        np.mean(
            np.asarray(
                values,
                dtype=np.float64,
            )
        )
    )


def metric_percentile(
    values,
    q,
):
    if not values:
        return None

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def print_capture_info(
    cap,
):
    actual_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    actual_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    actual_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    print(
        f"Capture device:     "
        f"{CAPTURE_DEVICE}"
    )

    print(
        f"Requested format:   "
        f"{CAPTURE_WIDTH}x"
        f"{CAPTURE_HEIGHT} "
        f"@ {CAPTURE_FPS:.3f} FPS"
    )

    print(
        f"OpenCV reports:     "
        f"{actual_width}x"
        f"{actual_height} "
        f"@ {actual_fps:.3f} FPS"
    )


def draw_prediction(
    frame,
    prediction,
):
    for (
        box,
        score,
        label,
    ) in zip(
        prediction["boxes"],
        prediction["scores"],
        prediction["labels"],
    ):
        x1, y1, x2, y2 = [
            int(v)
            for v in box
        ]

        name = LOCAL_TO_NAME[
            int(label)
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
            f"{name} {float(score):.2f}",
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


def show_frame(
    frame,
):
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


def run_demo(
    inference_target,
    output_path,
    result_metadata,
):
    # First open: acquire a real OBS frame for warmup.
    cap = open_capture()

    print_capture_info(
        cap
    )

    ok, warmup_frame = cap.read()

    if not ok:
        cap.release()
        raise RuntimeError(
            "Could not read a frame from OBS Virtual Camera."
        )

    inference_target(
        warmup_frame,
        warmup=True,
    )

    # Start deployment-like run from a fresh live feed.
    cap.release()
    cap = open_capture()

    capture_state = CaptureState()
    prediction_state = PredictionState()
    stats = InferenceStats()
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_worker,
        args=(
            cap,
            capture_state,
            stop_event,
        ),
        daemon=True,
    )

    def inference_worker():
        last_inferred_id = -1

        while not stop_event.is_set():
            item = capture_state.wait_for_newer(
                last_inferred_id
            )

            if item is None:
                break

            (
                frame_id,
                frame,
                arrival_time,
            ) = item

            last_inferred_id = frame_id

            torch.cuda.synchronize()
            predict_start = time.perf_counter()

            prediction = inference_target(
                frame,
                warmup=False,
            )

            torch.cuda.synchronize()
            predict_end = time.perf_counter()

            stats.predict_ms.append(
                (
                    predict_end
                    - predict_start
                )
                * 1000.0
            )

            stats.e2e_ms.append(
                (
                    predict_end
                    - arrival_time
                )
                * 1000.0
            )

            stats.detection_counts.append(
                len(
                    prediction[
                        "boxes"
                    ]
                )
            )

            stats.processed += 1

            prediction_state.publish(
                prediction,
                frame_id,
                arrival_time,
            )

    inference_thread = threading.Thread(
        target=inference_worker,
        daemon=True,
    )

    capture_thread.start()
    inference_thread.start()

    start_time = time.perf_counter()
    displayed = 0
    last_displayed_frame_id = -1
    detection_age_samples = []

    try:
        while True:
            now = time.perf_counter()

            if (
                RUN_SECONDS is not None
                and now - start_time
                >= RUN_SECONDS
            ):
                break

            item = capture_state.snapshot()

            if item is None:
                time.sleep(
                    0.001
                )
                continue

            (
                frame_id,
                frame,
                frame_arrival_time,
            ) = item

            if (
                frame_id
                == last_displayed_frame_id
            ):
                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key in (
                    ord("q"),
                    27,
                ):
                    break

                time.sleep(
                    0.001
                )
                continue

            last_displayed_frame_id = (
                frame_id
            )

            annotated = frame.copy()

            prediction_snapshot = (
                prediction_state.snapshot()
            )

            detection_age_ms = None
            prediction_visible = False

            if prediction_snapshot is not None:
                (
                    prediction,
                    prediction_frame_id,
                    prediction_source_time,
                    prediction_completed_time,
                ) = prediction_snapshot

                detection_age_ms = max(
                    0.0,
                    (
                        frame_arrival_time
                        - prediction_source_time
                    )
                    * 1000.0,
                )

                detection_age_samples.append(
                    detection_age_ms
                )

                prediction_visible = (
                    MAX_DETECTION_AGE_MS is None
                    or detection_age_ms
                    <= MAX_DETECTION_AGE_MS
                )

                if prediction_visible:
                    draw_prediction(
                        annotated,
                        prediction,
                    )

            elapsed = max(
                now - start_time,
                1e-9,
            )

            display_fps = (
                displayed
                / elapsed
            )

            inference_fps = (
                stats.processed
                / elapsed
            )

            if SHOW_DIAGNOSTICS:
                if detection_age_ms is None:
                    age_text = "no prediction yet"
                else:
                    age_text = (
                        f"boxes age "
                        f"{detection_age_ms:.0f} ms"
                    )

                cv2.putText(
                    annotated,
                    (
                        f"{DISPLAY_MODEL_NAME} | "
                        f"display {display_fps:.1f} FPS | "
                        f"infer {inference_fps:.1f} FPS | "
                        f"{age_text}"
                    ),
                    (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            show_frame(
                annotated
            )

            displayed += 1

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                ord("q"),
                27,
            ):
                break

    except KeyboardInterrupt:
        pass

    finally:
        stop_event.set()
        capture_state.finish()

        cap.release()

        capture_thread.join(
            timeout=2.0
        )

        inference_thread.join(
            timeout=5.0
        )

        cv2.destroyAllWindows()

    end_time = time.perf_counter()
    elapsed = max(
        end_time - start_time,
        1e-9,
    )

    captured = (
        capture_state.frames_captured
    )

    inferred = (
        stats.processed
    )

    skipped_for_inference = max(
        captured - inferred,
        0,
    )

    frame_deadline_ms = (
        1000.0
        / CAPTURE_FPS
    )

    e2e_array = np.asarray(
        stats.e2e_ms,
        dtype=np.float64,
    )

    row = {
        **result_metadata,

        "capture_device":
            CAPTURE_DEVICE,

        "requested_width":
            CAPTURE_WIDTH,

        "requested_height":
            CAPTURE_HEIGHT,

        "requested_fps":
            CAPTURE_FPS,

        "run_seconds":
            elapsed,

        "frames_captured":
            captured,

        "frames_displayed":
            displayed,

        "frames_inferred":
            inferred,

        "frames_skipped_for_inference":
            skipped_for_inference,

        "inference_skip_percent":
            (
                100.0
                * skipped_for_inference
                / captured
                if captured > 0
                else 0.0
            ),

        "capture_effective_fps":
            actual_capture_fps(
                capture_state
            ),

        "display_effective_fps":
            displayed
            / elapsed,

        "inference_effective_fps":
            inferred
            / elapsed,

        "predict_mean_ms":
            metric_mean(
                stats.predict_ms
            ),

        "predict_p95_ms":
            metric_percentile(
                stats.predict_ms,
                95,
            ),

        "predict_p99_ms":
            metric_percentile(
                stats.predict_ms,
                99,
            ),

        "e2e_mean_ms":
            metric_mean(
                stats.e2e_ms
            ),

        "e2e_p95_ms":
            metric_percentile(
                stats.e2e_ms,
                95,
            ),

        "e2e_p99_ms":
            metric_percentile(
                stats.e2e_ms,
                99,
            ),

        "deadline_miss_percent":
            (
                float(
                    100.0
                    * np.mean(
                        e2e_array
                        > frame_deadline_ms
                    )
                )
                if len(
                    e2e_array
                ) > 0
                else None
            ),

        "mean_detections_per_inferred_frame":
            metric_mean(
                stats.detection_counts
            ),

        "mean_displayed_detection_age_ms":
            metric_mean(
                detection_age_samples
            ),

        "p95_displayed_detection_age_ms":
            metric_percentile(
                detection_age_samples,
                95,
            ),

        "max_detection_age_ms":
            MAX_DETECTION_AGE_MS,
    }

    save_csv(
        output_path,
        row,
    )

    print()
    print("=" * 70)
    print("SMOOTH DEMO RESULTS")
    print("=" * 70)

    print(
        f"Capture FPS:          "
        f"{row['capture_effective_fps']:.2f}"
    )

    print(
        f"Display FPS:          "
        f"{row['display_effective_fps']:.2f}"
    )

    print(
        f"Inference FPS:        "
        f"{row['inference_effective_fps']:.2f}"
    )

    print(
        f"Inference skip:       "
        f"{row['inference_skip_percent']:.2f}%"
    )

    if row["e2e_mean_ms"] is not None:
        print(
            f"Inference E2E mean:   "
            f"{row['e2e_mean_ms']:.2f} ms"
        )

    if (
        row[
            "mean_displayed_detection_age_ms"
        ]
        is not None
    ):
        print(
            f"Mean box age:         "
            f"{row['mean_displayed_detection_age_ms']:.2f} ms"
        )

    print(
        f"Saved:                "
        f"{output_path}"
    )




def normalize_name(name):
    name = str(
        name
    ).lower()

    return MODEL_NAME_ALIASES.get(
        name,
        name,
    )


def get_starts(image_size):
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


def make_tiles(frame_bgr):
    height, width = (
        frame_bgr.shape[:2]
    )

    tiles = []
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

            tiles.append(
                tile_rgb
            )

            offsets.append(
                (x0, y0)
            )

    return (
        tiles,
        offsets,
    )


def rf_tiled_predict(
    frame_bgr,
    model,
):
    frame_h, frame_w = (
        frame_bgr.shape[:2]
    )

    (
        tiles,
        offsets,
    ) = make_tiles(
        frame_bgr
    )

    all_boxes = []
    all_scores = []
    all_labels = []

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

        detections_list = model.predict(
            batch_images,
            threshold=CONFIDENCE,
            include_source_image=False,
        )

        if not isinstance(
            detections_list,
            (list, tuple),
        ):
            if len(
                batch_images
            ) != 1:
                raise RuntimeError(
                    "RF-DETR batch prediction did "
                    "not return one result per tile."
                )

            detections_list = [
                detections_list
            ]

        for (
            detections,
            (x0, y0),
        ) in zip(
            detections_list,
            batch_offsets,
        ):
            if len(
                detections
            ) == 0:
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
        return {
            "boxes":
                np.empty(
                    (0, 4),
                    dtype=np.float32,
                ),

            "scores":
                np.empty(
                    (0,),
                    dtype=np.float32,
                ),

            "labels":
                np.empty(
                    (0,),
                    dtype=np.int64,
                ),
        }

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
        idx = torch.where(
            labels == class_id
        )[0]

        keep = nms(
            boxes[
                idx
            ],
            scores[
                idx
            ],
            NMS_IOU,
        )

        keep_all.append(
            idx[
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
            .numpy(),

        "scores":
            scores[
                keep
            ]
            .detach()
            .cpu()
            .numpy(),

        "labels":
            labels[
                keep
            ]
            .detach()
            .cpu()
            .numpy(),
    }



def output_path():
    confidence_tag = (
        f"{CONFIDENCE:.2f}"
        .replace(
            ".",
            "p",
        )
    )

    return (
        RESULTS_DIR
        / (
            f"{MODEL_NAME}"
            f"_tile{TILE_SIZE}"
            f"_conf{confidence_tag}"
            f"_smooth_demo.csv"
        )
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU is not available."
        )

    if MODEL_NAME not in MODELS:
        raise ValueError(
            f"Unknown MODEL_NAME: "
            f"{MODEL_NAME}"
        )

    model_class = MODELS[
        MODEL_NAME
    ]

    model = model_class(
        resolution=TILE_SIZE,
        device=DEVICE,
    )

    def inference_target(
        frame,
        warmup=False,
    ):
        if warmup:
            for _ in range(
                WARMUP_RUNS
            ):
                _ = rf_tiled_predict(
                    frame,
                    model,
                )

            torch.cuda.synchronize()

            return {
                "boxes":
                    np.empty(
                        (0, 4),
                        dtype=np.float32,
                    ),

                "scores":
                    np.empty(
                        (0,),
                        dtype=np.float32,
                    ),

                "labels":
                    np.empty(
                        (0,),
                        dtype=np.int64,
                    ),
            }

        return rf_tiled_predict(
            frame,
            model,
        )

    print("=" * 70)
    print(
        "RF-DETR SMOOTH OBS DEMO"
    )
    print("=" * 70)

    print(
        f"Model:              "
        f"{MODEL_NAME}"
    )

    print(
        f"Confidence:         "
        f"{CONFIDENCE}"
    )

    print(
        f"Tile size:          "
        f"{TILE_SIZE}"
    )

    print(
        f"Max box age:        "
        f"{MAX_DETECTION_AGE_MS} ms"
    )

    run_demo(
        inference_target=inference_target,
        output_path=output_path(),
        result_metadata={
            "model":
                MODEL_NAME,

            "method":
                "obs_smooth_display_tiled_batched",

            "tile_size":
                TILE_SIZE,

            "tile_overlap":
                TILE_OVERLAP,

            "tile_batch_size":
                TILE_BATCH_SIZE,

            "nms_iou":
                NMS_IOU,

            "confidence":
                CONFIDENCE,
        },
    )


if __name__ == "__main__":
    main()
