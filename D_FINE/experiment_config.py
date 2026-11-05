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


TILED_LATENCY_SUMMARY_FILE = (
    LATENCY_OUTPUT_DIR
    / "latencysummary_tiled_batched.csv"
)

TILED_ACCURACY_SUMMARY_FILE = (
    ACCURACY_OUTPUT_DIR
    / "accuracysummary_tiled_batched.csv"
)


# ============================================================
# D-FINE COCO MODELS
# ============================================================

MODELS = {
    "dfine_nano":
        "ustc-community/dfine-nano-coco",

    "dfine_small":
        "ustc-community/dfine-small-coco",

    "dfine_medium":
        "ustc-community/dfine-medium-coco",

    "dfine_large":
        "ustc-community/dfine-large-coco",

    "dfine_xlarge":
        "ustc-community/dfine-xlarge-coco",
}


# ============================================================
# STANDARDIZED RESOLUTIONS
# ============================================================

RESOLUTIONS = [
    640,
    960,
    1280,
    1920,
]


# ============================================================
# TILED INFERENCE
# ============================================================

TILE_SIZE = 640
TILE_OVERLAP = 0.20
TILED_NMS_IOU = 0.50

# 1920x1080 produces 8 tiles with this geometry.
# Batch all 8 together for one transformer forward pass.
TILE_BATCH_SIZE = 8


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda"


# ============================================================
# WARMUP
# ============================================================

WARMUP_RUNS = 20


# ============================================================
# LATENCY
# ============================================================

LATENCY_CONF = 0.25

SHOW_VIEWER = True

WINDOW_NAME = (
    "D-FINE Latency Test"
)


# ============================================================
# ACCURACY
# ============================================================

ACCURACY_CONF = 0.001


# ============================================================
# TARGET CLASSES
# ============================================================

TARGET_CLASS_NAMES = (
    "person",
    "car",
    "airplane",
)


# ============================================================
# MODEL-NAME NORMALIZATION
# ============================================================

MODEL_NAME_ALIASES = {
    "person": "person",
    "car": "car",

    # D-FINE COCO checkpoint spelling
    "aeroplane": "airplane",

    "airplane": "airplane",
}


# ============================================================
# COMMON EVALUATION SPACE
# ============================================================

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


# ============================================================
# MANUAL FPV GROUND TRUTH
# ============================================================

# Ground-truth files contain COCO IDs:
#
# 0 = person
# 2 = car
# 4 = airplane

GT_COCO_TO_LOCAL = {
    0: 0,
    2: 1,
    4: 2,
}