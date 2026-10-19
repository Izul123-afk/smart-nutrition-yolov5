"""Uji cepat model YOLOv8n (best2.pt / yolov8n .onnx) pada satu gambar.

YOLOv8 bersifat anchor-free: tidak ada objectness score terpisah
seperti YOLOv5. Tiap kandidat berisi [cx, cy, w, h, skor_per_kelas],
sehingga confidence = max(skor kelas).

Prioritas backend:
  1. File .pt  -> dipakai lewat API ultralytics (disarankan).
  2. File .onnx -> inferensi manual via onnxruntime dengan format
     output YOLOv8 [1, 4 + num_classes, N].

Contoh:
    python test.py
    python test.py --model best2.pt --image images.jpg
    python test.py --model best2.onnx --conf 0.25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def letterbox(image, target_size):
    """
    Resize gambar tanpa mengubah rasio asli.
    Area kosong diisi warna abu-abu seperti preprocessing Ultralytics.
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


def draw_detections(image, kept, class_names):
    result_image = image.copy()
    for index, (x, y, width, height, confidence, class_id) in enumerate(kept, start=1):
        if 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"class_{class_id}"
        label = f"{class_name} {confidence * 100:.1f}%"
        cv2.rectangle(result_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(
            result_image, label, (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
        )
        print(
            f"Deteksi {index}: {class_name}, "
            f"confidence={confidence * 100:.2f}%, "
            f"box=({x}, {y}, {width}, {height})"
        )
    return result_image


def run_ultralytics_pt(model_path, image_path, conf, iou, imgsz, output_path):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    class_names = [model.names[i] for i in sorted(model.names)]
    print("Jumlah kelas:", len(class_names))
    print("Daftar kelas:", class_names)

    start_time = time.perf_counter()
    results = model.predict(
        str(image_path), conf=conf, iou=iou, imgsz=imgsz,
        max_det=100, device="cpu", verbose=False,
    )[0]
    inference_ms = (time.perf_counter() - start_time) * 1000

    original_image = cv2.imread(str(image_path))
    boxes = results.boxes
    kept = []
    highest = 0.0
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            highest = max(highest, float(c))
            kept.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(c), int(k)))

    print(f"\nOutput ultralytics: {len(kept)} box (NMS internal YOLOv8)")
    print(f"Waktu inference: {inference_ms:.2f} ms")
    result_image = draw_detections(original_image, kept, class_names)
    cv2.imwrite(str(output_path), result_image)
    return len(kept), highest


def run_onnx_yolov8(model_path, image_path, classes_path, conf, iou, output_path):
    import onnxruntime as ort

    with classes_path.open("r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file if line.strip()]
    if not class_names:
        raise RuntimeError("classes.txt kosong.")
    print("Jumlah kelas:", len(class_names))
    print("Daftar kelas:", class_names)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 2
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape
    input_height = input_shape[2] if isinstance(input_shape[2], int) else 640
    input_width = input_shape[3] if isinstance(input_shape[3], int) else 640
    print("\nModel ONNX YOLOv8 berhasil dimuat")
    print("Input name :", input_name)
    print("Input shape:", input_shape)
    print("Ukuran input:", input_width, "x", input_height)

    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise RuntimeError("Gambar gagal dibaca oleh OpenCV.")
    original_height, original_width = original_image.shape[:2]
    prepared_image, scale, pad_x, pad_y = letterbox(
        original_image, (input_height, input_width))
    rgb_image = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2RGB)
    input_tensor = (rgb_image.transpose(2, 0, 1).astype(np.float32) / 255.0)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    start_time = time.perf_counter()
    outputs = session.run(None, {input_name: input_tensor})
    inference_ms = (time.perf_counter() - start_time) * 1000
    predictions = outputs[0]
    print("\nOutput ONNX:", predictions.shape)
    print(f"Waktu inference: {inference_ms:.2f} ms")

    # Output YOLOv8: [1, 4 + num_classes, N] (cx, cy, w, h + skor kelas).
    if predictions.ndim == 3:
        predictions = predictions[0]  # [4 + nc, N]
    if predictions.ndim != 2:
        raise RuntimeError(f"Bentuk output model tidak didukung: {predictions.shape}")
    num_classes = predictions.shape[0] - 4
    if num_classes < 1:
        # Kemungkinan format lama [N, 4 + nc] — transpose agar konsisten.
        predictions = predictions.transpose()
        num_classes = predictions.shape[1] - 4
    if num_classes != len(class_names):
        print(f"Peringatan: jumlah kelas model ({num_classes}) != classes.txt ({len(class_names)}).")

    boxes, scores, class_ids = [], [], []
    highest_confidence = 0.0
    num_preds = predictions.shape[1]
    for i in range(num_preds):
        col = predictions[:, i]
        cx, cy, w, h = (float(v) for v in col[:4])
        class_scores = col[4:4 + len(class_names)]
        if len(class_scores) == 0:
            continue
        class_id = int(np.argmax(class_scores))
        confidence = float(class_scores[class_id])
        highest_confidence = max(highest_confidence, confidence)
        if confidence < conf:
            continue
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        x1 = int((x1 - pad_x) / scale)
        y1 = int((y1 - pad_y) / scale)
        x2 = int((x2 - pad_x) / scale)
        y2 = int((y2 - pad_y) / scale)
        x1 = max(0, min(x1, original_width - 1))
        y1 = max(0, min(y1, original_height - 1))
        x2 = max(0, min(x2, original_width - 1))
        y2 = max(0, min(y2, original_height - 1))
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            continue
        boxes.append([x1, y1, width, height])
        scores.append(confidence)
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    kept = []
    if len(indices) > 0:
        for index in np.array(indices).reshape(-1):
            x, y, width, height = boxes[index]
            kept.append((x, y, width, height, scores[index], class_ids[index]))
    result_image = draw_detections(original_image, kept, class_names)
    cv2.imwrite(str(output_path), result_image)
    return len(kept), highest_confidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Uji model YOLOv8n pada satu gambar.")
    parser.add_argument("--model", default="best2.pt", help="Path model .pt (YOLOv8n) atau .onnx")
    parser.add_argument("--image", default="images.jpg", help="Gambar uji")
    parser.add_argument("--classes", default="classes.txt", help="Daftar kelas (khusus backend ONNX)")
    parser.add_argument("--output", default="hasil_uji.jpg", help="Gambar hasil anotasi")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    output_path = Path(args.output)
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path.resolve()}")
    if not image_path.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path.resolve()}")

    if model_path.suffix.lower() == ".pt":
        count, highest = run_ultralytics_pt(
            model_path, image_path, args.conf, args.iou, args.imgsz, output_path)
    elif model_path.suffix.lower() == ".onnx":
        count, highest = run_onnx_yolov8(
            model_path, image_path, Path(args.classes), args.conf, args.iou, output_path)
    else:
        raise SystemExit("Format model harus .pt atau .onnx.")

    print("\n===================================")
    print("HASIL PENGUJIAN (YOLOv8n)")
    print("===================================")
    print("Jumlah deteksi       :", count)
    print("Confidence tertinggi:", f"{highest * 100:.2f}%")
    print("Hasil gambar         :", output_path.resolve())
    if count == 0:
        print("\nModel berjalan, tetapi tidak ada objek yang melewati threshold.")
        print("Coba gunakan gambar asli dari dataset validasi.")
        print("Untuk diagnosis, threshold dapat diturunkan sementara menjadi 0.10.")
    else:
        print("\nModel YOLOv8n berhasil mendeteksi objek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
