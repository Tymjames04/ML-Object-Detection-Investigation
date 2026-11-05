from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent

VIDEO_PATH = (
    ROOT_DIR
    / "videos"
    / "30fps.mp4"
)

ACCURACY_IMAGES_DIR = (
    ROOT_DIR
    / "accuracy_test"
    / "images"
)

ACCURACY_LABELS_DIR = (
    ROOT_DIR
    / "accuracy_test"
    / "labels"
)

RESULTS_DIR = (
    ROOT_DIR
    / "results"
)

LATENCY_OUTPUT_DIR = (
    RESULTS_DIR
    / "video_latency"
)

LATENCY_SUMMARY_FILE = (
    LATENCY_OUTPUT_DIR
    / "latencysummary.csv"
)

ACCURACY_OUTPUT_DIR = (
    RESULTS_DIR
    / "image_accuracy"
)

ACCURACY_SUMMARY_FILE = (
    ACCURACY_OUTPUT_DIR
    / "accuracysummary.csv"
)


# ============================================================
# MODELS
# ============================================================

MODELS = [
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
]


# ============================================================
# RESOLUTIONS
# ============================================================

RESOLUTIONS = [
    640,
    960,
    1280,
    1920,
]


# ============================================================
# DEVICE / MODEL SETTINGS
# ============================================================

DEVICE = 0

# Explicit FP32, matching the existing YOLO benchmark.
QUANTIZE = 32

# Preserve source aspect ratio with minimal stride padding.
RECT = True


# ============================================================
# WARMUP / VIEWER
# ============================================================

WARMUP_RUNS = 20

# Keep False for recorded benchmark numbers.
SHOW_VIEWER = False

WINDOW_NAME = "YOLO26 Latency Test"


# ============================================================
# CONFIDENCE
# ============================================================

LATENCY_CONF = 0.25
ACCURACY_CONF = 0.001


# ============================================================
# THREE-CLASS EVALUATION SPACE
# ============================================================

# COCO IDs used by both the pretrained YOLO model and the
# manually reviewed FPV label files.
COCO_EVAL_CLASSES = [
    0,  # person
    2,  # car
    4,  # airplane
]

COCO_TO_LOCAL = {
    0: 0,  # person
    2: 1,  # car
    4: 2,  # airplane
}

LOCAL_TO_NAME = {
    0: "person",
    1: "car",
    2: "airplane",
}
