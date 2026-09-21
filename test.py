from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


# =========================================================
# KONFIGURASI
# =========================================================

MODEL_PATH = Path("best.onnx")
IMAGE_PATH = Path("images.jpg")
CLASSES_PATH = Path("classes.txt")
OUTPUT_PATH = Path("hasil_uji.jpg")

CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45


# =========================================================
# LETTERBOX: MEMPERTAHANKAN RASIO GAMBAR
# =========================================================

def letterbox(image, target_size):
    """
    Resize gambar tanpa mengubah rasio asli.
    Area kosong diisi warna abu-abu seperti preprocessing YOLOv5.
    """

    target_height, target_width = target_size
    original_height, original_width = image.shape[:2]

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height

    left = int(round(horizontal_padding / 2 - 0.1))
    right = int(round(horizontal_padding / 2 + 0.1))
    top = int(round(vertical_padding / 2 - 0.1))
    bottom = int(round(vertical_padding / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    return padded, scale, left, top


# =========================================================
# VALIDASI FILE
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH.resolve()}")

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH.resolve()}")

if not CLASSES_PATH.exists():
    raise FileNotFoundError(
        f"Daftar kelas tidak ditemukan: {CLASSES_PATH.resolve()}"
    )


# =========================================================
# MEMBACA NAMA KELAS
# =========================================================

with CLASSES_PATH.open("r", encoding="utf-8") as file:
    class_names = [
        line.strip()
        for line in file
        if line.strip()
    ]

if not class_names:
    raise RuntimeError("classes.txt kosong.")

print("Jumlah kelas:", len(class_names))
print("Daftar kelas:", class_names)


# =========================================================
# MEMUAT MODEL ONNX
# =========================================================

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 2
session_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=session_options,
    providers=["CPUExecutionProvider"],
)

input_info = session.get_inputs()[0]
input_name = input_info.name
input_shape = input_info.shape

# Umumnya YOLOv5 memakai bentuk [1, 3, 640, 640].
input_height = (
    input_shape[2]
    if isinstance(input_shape[2], int)
    else 640
)

input_width = (
    input_shape[3]
    if isinstance(input_shape[3], int)
    else 640
)

print("\nModel berhasil dimuat")
print("Input name :", input_name)
print("Input shape:", input_shape)
print("Ukuran input:", input_width, "x", input_height)


# =========================================================
# MEMBACA DAN MENYIAPKAN GAMBAR
# =========================================================

original_image = cv2.imread(str(IMAGE_PATH))

if original_image is None:
    raise RuntimeError("Gambar gagal dibaca oleh OpenCV.")

original_height, original_width = original_image.shape[:2]

prepared_image, scale, pad_x, pad_y = letterbox(
    original_image,
    (input_height, input_width),
)

# OpenCV BGR → RGB
rgb_image = cv2.cvtColor(
    prepared_image,
    cv2.COLOR_BGR2RGB,
)

# HWC → CHW, normalisasi 0–1, dan tambah batch
input_tensor = (
    rgb_image
    .transpose(2, 0, 1)
    .astype(np.float32)
    / 255.0
)

input_tensor = np.expand_dims(
    input_tensor,
    axis=0,
)


# =========================================================
# INFERENCE
# =========================================================

start_time = time.perf_counter()

outputs = session.run(
    None,
    {input_name: input_tensor},
)

inference_time_ms = (
    time.perf_counter() - start_time
) * 1000

predictions = outputs[0]

print("\nOutput ONNX:", predictions.shape)
print(f"Waktu inference: {inference_time_ms:.2f} ms")

# Umumnya output YOLOv5: [1, jumlah_prediksi, 5 + jumlah_kelas]
if predictions.ndim == 3:
    predictions = predictions[0]

# Menangani output yang mungkin berbentuk [fitur, jumlah_prediksi].
expected_columns = 5 + len(class_names)

if (
    predictions.ndim == 2
    and predictions.shape[0] == expected_columns
    and predictions.shape[1] != expected_columns
):
    predictions = predictions.transpose()

print("Output setelah diproses:", predictions.shape)

if predictions.ndim != 2:
    raise RuntimeError(
        f"Bentuk output model tidak didukung: {predictions.shape}"
    )

if predictions.shape[1] < 6:
    raise RuntimeError(
        "Output model tidak mempunyai data deteksi yang lengkap."
    )


# =========================================================
# MEMBACA PREDIKSI YOLOV5
# =========================================================

boxes = []
scores = []
class_ids = []

highest_confidence = 0.0

for detection in predictions:
    objectness = float(detection[4])
    class_scores = detection[5:]

    if len(class_scores) == 0:
        continue

    class_id = int(np.argmax(class_scores))
    class_probability = float(class_scores[class_id])

    # Confidence final YOLOv5
    confidence = objectness * class_probability

    highest_confidence = max(
        highest_confidence,
        confidence,
    )

    if confidence < CONFIDENCE_THRESHOLD:
        continue

    center_x, center_y, box_width, box_height = detection[:4]

    # Koordinat pada gambar letterbox
    x1 = center_x - box_width / 2
    y1 = center_y - box_height / 2
    x2 = center_x + box_width / 2
    y2 = center_y + box_height / 2

    # Hilangkan padding, lalu kembali ke ukuran gambar asli
    x1 = int((x1 - pad_x) / scale)
    y1 = int((y1 - pad_y) / scale)
    x2 = int((x2 - pad_x) / scale)
    y2 = int((y2 - pad_y) / scale)

    # Batasi koordinat agar tidak keluar gambar
    x1 = max(0, min(x1, original_width - 1))
    y1 = max(0, min(y1, original_height - 1))
    x2 = max(0, min(x2, original_width - 1))
    y2 = max(0, min(y2, original_height - 1))

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        continue

    boxes.append([x1, y1, width, height])
    scores.append(confidence)
    class_ids.append(class_id)


# =========================================================
# NON-MAXIMUM SUPPRESSION
# =========================================================

indices = cv2.dnn.NMSBoxes(
    boxes,
    scores,
    CONFIDENCE_THRESHOLD,
    NMS_THRESHOLD,
)

result_image = original_image.copy()
detection_count = 0

if len(indices) > 0:
    selected_indices = np.array(indices).reshape(-1)

    for index in selected_indices:
        x, y, width, height = boxes[index]
        confidence = scores[index]
        class_id = class_ids[index]

        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"

        label = f"{class_name} {confidence * 100:.1f}%"

        cv2.rectangle(
            result_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )

        text_y = max(y - 10, 25)

        cv2.putText(
            result_image,
            label,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        detection_count += 1

        print(
            f"Deteksi {detection_count}: "
            f"{class_name}, "
            f"confidence={confidence * 100:.2f}%, "
            f"box=({x}, {y}, {width}, {height})"
        )


# =========================================================
# SIMPAN HASIL
# =========================================================

cv2.imwrite(
    str(OUTPUT_PATH),
    result_image,
)

print("\n===================================")
print("HASIL PENGUJIAN")
print("===================================")
print("Jumlah deteksi       :", detection_count)
print(
    "Confidence tertinggi:",
    f"{highest_confidence * 100:.2f}%",
)
print("Hasil gambar         :", OUTPUT_PATH.resolve())

if detection_count == 0:
    print("\nModel berjalan, tetapi tidak ada objek yang melewati threshold.")
    print("Coba gunakan gambar asli dari dataset validasi.")
    print("Untuk diagnosis, threshold dapat diturunkan sementara menjadi 0.10.")
else:
    print("\nModel ONNX berhasil mendeteksi objek.")