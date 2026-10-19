from __future__ import annotations

from pathlib import Path

import threading
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class YOLOService:
    """
    Service inferensi model deteksi YOLOv8n (Ultralytics).

    Checkpoint ``best2.pt`` adalah model YOLOv8n hasil training
    (.pt Ultralytics), sehingga dimuat lewat API ``ultralytics.YOLO``.
    Model menggunakan PyTorch CPU dan hanya dijalankan saat tombol
    scan ditekan (single-shot inference).

    Berbeda dari YOLOv5, YOLOv8 bersifat anchor-free tanpa objectness
    score terpisah — skor class langsung menjadi confidence. Seluruh
    decoding + NMS ditangani oleh ``model.predict()``, service ini
    hanya memformat hasilnya ke dict standar aplikasi.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        max_detections: int = 1,
        torch_threads: int = 2,
        class_confidences: dict[str, float] | None = None,
    ) -> None:

        self.model_path = Path(
            model_path
        ).resolve()

        self.confidence = confidence

        # Threshold per-kelas, mis. {"bakso": 0.20}. Key dinormalisasi
        # ke lowercase agar pencocokan nama kelas tidak case-sensitive.
        self.class_confidences: dict[str, float] = {}
        if class_confidences:
            for name, value in class_confidences.items():
                try:
                    self.class_confidences[str(name).lower()] = float(value)
                except (TypeError, ValueError):
                    continue

        self.iou_threshold = iou_threshold

        self.max_detections = max_detections

        self.image_size = image_size


        self._lock = threading.Lock()

        self._last_error: str | None = None


        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model YOLOv8n tidak ditemukan: "
                f"{self.model_path}"
            )


        # Batasi penggunaan CPU Raspberry Pi
        torch.set_num_threads(
            max(1, torch_threads)
        )


        try:

            torch.set_num_interop_threads(1)

        except RuntimeError:

            pass


        print(
            f"Memuat model YOLOv8n: {self.model_path.name}..."
        )


        # Checkpoint YOLOv8n Ultralytics, dimuat lewat API resminya
        self.model = YOLO(
            str(self.model_path)
        )


        # Nama kelas langsung dari checkpoint (model.names)
        self.class_names = (
            self.model.names
        )


        self._loaded = True


        print(
            "Model YOLOv8n berhasil dimuat."
        )

        print(
            f"Nama kelas dari {self.model_path.name}:",
            self.class_names,
        )


    # =====================================================
    # MENGAMBIL NAMA KELAS
    # =====================================================

    def _get_class_name(
        self,
        class_id: int,
    ) -> str:

        if isinstance(
            self.class_names,
            dict,
        ):

            return str(

                self.class_names.get(

                    class_id,

                    f"class_{class_id}",

                )

            )


        if (
            0
            <= class_id
            < len(self.class_names)
        ):

            return str(
                self.class_names[class_id]
            )


        return f"class_{class_id}"


    def _threshold_for(self, class_name: str) -> float:
        """Threshold conf untuk satu kelas (0-1). Default = self.confidence."""
        return self.class_confidences.get(str(class_name).lower(), self.confidence)

    def _effective_confidence(self) -> float:
        """Threshold terendah yang diteruskan ke model.predict()."""
        if not self.class_confidences:
            return self.confidence
        return min([self.confidence, *self.class_confidences.values()])

    # =====================================================
    # INFERENCE (YOLOv8n anchor-free via ultralytics)
    # =====================================================

    def predict(
        self,
        frame: np.ndarray,
    ) -> dict:

        if (
            frame is None
            or frame.size == 0
        ):

            raise ValueError(
                "Frame kamera kosong."
            )


        original_frame = frame.copy()


        with self._lock:

            started_at = (
                time.perf_counter()
            )


            try:

                # ultralytics menerima numpy BGR dan mengonversinya
                # ke RGB sendiri, jadi frame diteruskan apa adanya.
                # imgsz YOLOv8n default 640 (bisa 320/416 untuk Pi lambat).
                # Pakai threshold terendah agar kelas yang diturunkan
                # (mis. bakso 0.20) tetap lolos dari model, lalu saring
                # per-kelas di bawah setelah NMS.
                results = self.model.predict(

                    original_frame,

                    conf=self._effective_confidence(),

                    iou=self.iou_threshold,

                    imgsz=self.image_size,

                    max_det=self.max_detections,

                    device="cpu",

                    verbose=False,

                )[0]


                inference_ms = (

                    time.perf_counter()
                    - started_at

                ) * 1000


                # Format ultralytics (sudah termasuk NMS):
                # x1, y1, x2, y2, confidence, class_id
                # confidence YOLOv8 = skor class langsung (tanpa objectness).
                boxes = results.boxes

                if boxes is None or len(boxes) == 0:
                    predictions = np.empty((0, 6))
                else:
                    predictions = np.column_stack(
                        (
                            boxes.xyxy.cpu().numpy(),
                            boxes.conf.cpu().numpy(),
                            boxes.cls.cpu().numpy(),
                        )
                    )

                detections: list[dict] = []

                for prediction in predictions:
                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        class_id,
                    ) = prediction.tolist()

                    class_id = int(class_id)
                    confidence = float(confidence)

                    class_name = self._get_class_name(class_id)

                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))

                    detection_data = {
                        "class_id": class_id,
                        "class_name": (class_name),
                        "confidence": round(
                            confidence * 100,
                            2,
                        ),
                        "box": {
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        },
                    }

                    # Saring per-kelas: bakso boleh lolos di threshold
                    # yang lebih rendah, kelas lain tetap di default.
                    if confidence + 1e-9 < self._threshold_for(class_name):
                        continue

                    detections.append(detection_data)

                # Confidence tertinggi di urutan pertama,
                # lalu HANYA ambil top-N sesuai konfigurasi.
                detections.sort(
                    key=lambda item: (item["confidence"]),
                    reverse=True,
                )
                detections = detections[: max(1, self.max_detections)]

                annotated_frame = original_frame.copy()

                for detection_data in detections:
                    box = detection_data["box"]
                    x1, y1 = box["x1"], box["y1"]
                    x2, y2 = box["x2"], box["y2"]
                    label = (
                        f"{detection_data['class_name']} "
                        f"{detection_data['confidence']:.1f}%"
                    )

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (
                            x1,
                            max(25, y1 - 10),
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )


                if detections:

                    detected = True

                    best_detection = (
                        detections[0]
                    )

                    food_name = (
                        best_detection[
                            "class_name"
                        ]
                    )

                    best_confidence = (
                        best_detection[
                            "confidence"
                        ]
                    )


                else:

                    detected = False

                    food_name = (
                        "Tidak terdeteksi"
                    )

                    best_confidence = 0.0


                self._last_error = None


                return {

                    "detected": detected,

                    "food_name": food_name,

                    "confidence": (
                        best_confidence
                    ),

                    "object_count": len(
                        detections
                    ),

                    "inference_ms": round(
                        inference_ms,
                        2,
                    ),

                    "detections": detections,

                    "annotated_frame": (
                        annotated_frame
                    ),

                }


            except Exception as error:

                self._last_error = str(
                    error
                )

                raise


    # =====================================================
    # STATUS MODEL
    # =====================================================

    def get_status(self) -> dict:

        return {

            "loaded": self._loaded,

            "error": self._last_error,

            "model": (
                self.model_path.name
            ),

            "backend": (
                "Ultralytics YOLOv8n PyTorch CPU"
            ),

            "image_size": (
                self.image_size
            ),

            "confidence": (
                self.confidence
            ),

            "class_confidences": (
                dict(self.class_confidences)
            ),

            "classes": (
                self.class_names
            ),

        }
