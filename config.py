from pathlib import Path


# =========================================================
# LOKASI DIREKTORI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best.pt"

_local_yolo = BASE_DIR / "yolov5"
_doc_yolo = Path.home() / "Documents" / "yolov5"
YOLOV5_REPO = _local_yolo if _local_yolo.exists() else _doc_yolo

TEST_IMAGE_PATH = BASE_DIR / "images.jpg"

DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "food_recognition.db"
PHOTO_DIR = DATA_DIR / "photos"
NUTRITION_CSV_PATH = BASE_DIR / "csv.csv"

# Folder lama tetap dibaca sekali untuk mengimpor foto hasil sebelumnya.
RESULT_DIR = (
    BASE_DIR
    / "static"
    / "results"
)


# =========================================================
# KONFIGURASI KAMERA
# =========================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_FPS = 30

CAMERA_JPEG_QUALITY = 80


# =========================================================
# KONFIGURASI YOLOv5
# =========================================================

YOLO_CONFIDENCE = 0.25

YOLO_IOU_THRESHOLD = 0.45

YOLO_IMAGE_SIZE = 640

YOLO_MAX_DETECTIONS = 10

# Raspberry Pi 3 jangan terlalu banyak thread
TORCH_THREADS = 2


# =========================================================
# KONFIGURASI HX711
# =========================================================

HX711_DT_PIN = 5

HX711_SCK_PIN = 6

# Faktor kalibrasi yang diperoleh sebelumnya
HX711_CALIBRATION_FACTOR = 718.42

HX711_RAW_SAMPLES = 5

HX711_TARE_SAMPLES = 20

HX711_FILTER_SIZE = 10

# Pembacaan hardware tidak boleh menahan startup Flask tanpa batas.
HX711_READ_TIMEOUT_SECONDS = 3.0
HX711_RECONNECT_INTERVAL_SECONDS = 3.0
HX711_LOG_INTERVAL_SECONDS = 30.0


# =========================================================
# KONFIGURASI FLASK
# =========================================================

HOST = "0.0.0.0"

PORT = 5000
