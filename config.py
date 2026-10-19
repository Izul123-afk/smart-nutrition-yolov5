from pathlib import Path


# =========================================================
# LOKASI DIREKTORI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "best2.pt"

TEST_IMAGE_PATH = BASE_DIR / "images.jpg"

DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "food_recognition.db"
PHOTO_DIR = DATA_DIR / "photos"
EXPORT_DIR = DATA_DIR / "exports"
# Satu file CSV hasil untuk scan manual: selalu ditulis ulang utuh dan
# urut per class setiap ada update (hanya dari scan yang di-export ke PDF).
HASIL_CSV_PATH = EXPORT_DIR / "hasil.csv"
# File lama (hasil_scan.csv): isinya dimigrasi otomatis ke hasil.csv
# lalu file lama dihapus.
LEGACY_CSV_PATH = EXPORT_DIR / "hasil_scan.csv"
NUTRITION_CSV_PATH = BASE_DIR / "csv.csv"

# Folder gambar uji untuk scan manual lewat terminal (manual_scan.py).
MANUAL_IMAGE_DIR = BASE_DIR / "gambar"

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
# KONFIGURASI YOLOv8n (Ultralytics)
# =========================================================

YOLO_CONFIDENCE = 0.25

# Threshold per-kelas (opsional). Kelas yang tidak tercantum memakai
# YOLO_CONFIDENCE. Bakso, capcay kuah, dan nasi goreng sedikit diturunkan
# agar lebih mudah terdeteksi.
YOLO_CLASS_CONFIDENCE = {
    "bakso": 0.20,
    "capcay kuah": 0.15,
    "nasi goreng": 0.15,
}

YOLO_IOU_THRESHOLD = 0.45

# Resolusi input default YOLOv8n. Turunkan ke 320/416 bila perlu
# inference lebih cepat di Raspberry Pi 3.
YOLO_IMAGE_SIZE = 320

YOLO_MAX_DETECTIONS = 1

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
