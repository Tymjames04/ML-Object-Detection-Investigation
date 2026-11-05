from pathlib import Path
import csv
import threading
import time
import os
import cv2
import numpy as np
import torch
from torchvision.ops import nms
from transformers import AutoImageProcessor, DFineForObjectDetection


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

# Set to None to run until q / Esc / Ctrl+C.
RUN_SECONDS = None

MODEL_NAME = "dfine_xlarge"

MODELS = {
    "dfine_nano": "ustc-community/dfine-nano-coco",
    "dfine_small": "ustc-community/dfine-small-coco",
    "dfine_medium": "ustc-community/dfine-medium-coco",
    "dfine_large": "ustc-community/dfine-large-coco",
    "dfine_xlarge": "ustc-community/dfine-xlarge-coco",
}

TILE_SIZE = 640
TILE_OVERLAP = 0.20
TILE_BATCH_SIZE = 8
NMS_IOU = 0.50

CONFIDENCE = 0.25

WARMUP_RUNS = 20
DEVICE = "cuda"

# Smooth demo behavior:
# The raw video remains live at capture FPS even when inference is slow.
# Latest detections are reused between inference updates.
MAX_DETECTION_AGE_MS = 500.0

SHOW_DIAGNOSTICS = True
DISPLAY_SCALE = 1.0

WINDOW_NAME = "D-FINE Smooth OBS Demo"

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
    frame_h, frame_w = (
        frame_bgr.shape[:2]
    )

    tiles = []
    offsets = []
    sizes = []

    for y0 in get_starts(
        frame_h
    ):
        for x0 in get_starts(
            frame_w
        ):
            tile_bgr = frame_bgr[
                y0:y0 + TILE_SIZE,
                x0:x0 + TILE_SIZE,
            ]

            tile_rgb = cv2.cvtColor(
                tile_bgr,
                cv2.COLOR_BGR2RGB,
            )

            tile_h, tile_w = (
                tile_rgb.shape[:2]
            )

            tiles.append(
                tile_rgb
            )

            offsets.append(
                (x0, y0)
            )

            sizes.append(
                (tile_h, tile_w)
            )

    return (
        tiles,
        offsets,
        sizes,
    )


def build_class_map(model):
    mapping = {}

    for (
        label_id,
        raw_name,
    ) in model.config.id2label.items():

        name = normalize_name(
            raw_name
        )

        if name in TARGET_CLASS_NAMES:
            mapping[
                int(label_id)
            ] = NAME_TO_LOCAL[
                name
            ]

    if not mapping:
        raise RuntimeError(
            "No target classes matched "
            "model.config.id2label."
        )

    return mapping


def hf_tiled_predict(
    frame_bgr,
    processor,
    model,
    class_map,
):
    frame_h, frame_w = (
        frame_bgr.shape[:2]
    )

    (
        tiles,
        offsets,
        sizes,
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

        batch_sizes = (
            sizes[start:end]
        )

        inputs = processor(
            images=batch_images,
            size={
                "height": TILE_SIZE,
                "width": TILE_SIZE,
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

        with torch.inference_mode():
            outputs = model(
                **inputs
            )

        target_sizes = torch.tensor(
            batch_sizes,
            dtype=torch.int64,
            device=DEVICE,
        )

        predictions = (
            processor
            .post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=CONFIDENCE,
            )
        )

        for (
            prediction,
            (x0, y0),
        ) in zip(
            predictions,
            batch_offsets,
        ):
            boxes = prediction[
                "boxes"
            ].to(
                torch.float32
            )

            scores = prediction[
                "scores"
            ].to(
                torch.float32
            )

            model_labels = prediction[
                "labels"
            ].to(
                torch.int64
            )

            for (
                model_label_id,
                local_label,
            ) in class_map.items():

                mask = (
                    model_labels
                    == model_label_id
                )

                if not torch.any(mask):
                    continue

                class_boxes = (
                    boxes[
                        mask
                    ].clone()
                )

                class_scores = (
                    scores[
                        mask
                    ]
                )

                class_boxes[:, 0] += x0
                class_boxes[:, 2] += x0
                class_boxes[:, 1] += y0
                class_boxes[:, 3] += y0

                class_boxes[:, 0].clamp_(
                    0,
                    frame_w,
                )

                class_boxes[:, 2].clamp_(
                    0,
                    frame_w,
                )

                class_boxes[:, 1].clamp_(
                    0,
                    frame_h,
                )

                class_boxes[:, 3].clamp_(
                    0,
                    frame_h,
                )

                valid = (
                    (
                        class_boxes[:, 2]
                        > class_boxes[:, 0]
                    )
                    &
                    (
                        class_boxes[:, 3]
                        > class_boxes[:, 1]
                    )
                )

                class_boxes = (
                    class_boxes[
                        valid
                    ]
                )

                class_scores = (
                    class_scores[
                        valid
                    ]
                )

                if (
                    class_boxes.shape[0]
                    == 0
                ):
                    continue

                class_labels = torch.full(
                    (
                        class_boxes.shape[
                            0
                        ],
                    ),
                    local_label,
                    dtype=torch.int64,
                    device=class_boxes.device,
                )

                all_boxes.append(
                    class_boxes
                )

                all_scores.append(
                    class_scores
                )

                all_labels.append(
                    class_labels
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

    boxes = torch.cat(
        all_boxes,
        dim=0,
    )

    scores = torch.cat(
        all_scores,
        dim=0,
    )

    labels = torch.cat(
        all_labels,
        dim=0,
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

    model_id = MODELS[
        MODEL_NAME
    ]

    processor = (
        AutoImageProcessor.from_pretrained(model_id)
    )

    model = (
        DFineForObjectDetection.from_pretrained(model_id)
        .to(
            DEVICE
        )
        .eval()
    )

    class_map = build_class_map(
        model
    )

    def inference_target(
        frame,
        warmup=False,
    ):
        if warmup:
            for _ in range(
                WARMUP_RUNS
            ):
                _ = hf_tiled_predict(
                    frame,
                    processor,
                    model,
                    class_map,
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

        return hf_tiled_predict(
            frame,
            processor,
            model,
            class_map,
        )

    print("=" * 70)
    print("D-FINE SMOOTH OBS DEMO")
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

    print(
        "Controls:           "
        "q / Esc = quit"
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
