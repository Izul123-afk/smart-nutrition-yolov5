from __future__ import annotations

from pathlib import Path

import threading
import time

import cv2
import numpy as np
import torch

from services.label_mapping import get_display_label


class YOLOService:
    """
    Service inferensi YOLOv5 best.pt.

    Model menggunakan PyTorch CPU.
    Model hanya dijalankan ketika tombol scan ditekan.
    """

    def __init__(
        self,
        model_path: str | Path,
        yolov5_path: str | Path,
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        max_detections: int = 10,
        torch_threads: int = 2,
    ) -> None:

        self.model_path = Path(
            model_path
        ).resolve()

        self.yolov5_path = Path(
            yolov5_path
        ).resolve()

        self.image_size = image_size


        self._lock = threading.Lock()

        self._last_error: str | None = None


        if not self.model_path.exists():

            raise FileNotFoundError(
                "best.pt tidak ditemukan: "
                f"{self.model_path}"
            )


        if not self.yolov5_path.exists():

            raise FileNotFoundError(
                "Folder YOLOv5 tidak ditemukan: "
                f"{self.yolov5_path}"
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
            "Memuat model YOLOv5 best.pt..."
        )


        # Memuat model dari repository YOLOv5 lokal
        self.model = torch.hub.load(

            str(self.yolov5_path),

            "custom",

            path=str(self.model_path),

            source="local",

            force_reload=False,

            verbose=False,

        )


        self.model.to("cpu")

        self.model.eval()


        self.model.conf = confidence

        self.model.iou = iou_threshold

        self.model.max_det = (
            max_detections
        )


        # Jangan gunakan AMP pada Raspberry Pi CPU
        self.model.amp = False


        # Nama kelas langsung dari best.pt
        self.class_names = (
            self.model.names
        )


        self._loaded = True


        print(
            "Model YOLOv5 berhasil dimuat."
        )

        print(
            "Nama kelas dari best.pt:",
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

            model_class_name = str(

                self.class_names.get(

                    class_id,

                    f"class_{class_id}",

                )

            )

            return get_display_label(
                model_class_name
            )


        if (
            0
            <= class_id
            < len(self.class_names)
        ):

            model_class_name = str(
                self.class_names[class_id]
            )

            return get_display_label(
                model_class_name
            )


        return f"class_{class_id}"


    # =====================================================
    # INFERENCE
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


        # Model AutoShape YOLOv5 dapat menerima numpy
        # tetapi RGB lebih konsisten dengan proses detect.py
        rgb_frame = cv2.cvtColor(

            original_frame,

            cv2.COLOR_BGR2RGB,

        )


        with self._lock:

            started_at = (
                time.perf_counter()
            )


            try:

                with torch.inference_mode():

                    results = self.model(

                        rgb_frame,

                        size=self.image_size,

                    )


                inference_ms = (

                    time.perf_counter()
                    - started_at

                ) * 1000


                # Format:
                # x1, y1, x2, y2, confidence, class_id
                predictions = (

                    results.xyxy[0]

                    .detach()

                    .cpu()

                    .numpy()

                )


                detections: list[dict] = []


                annotated_frame = (
                    original_frame.copy()
                )


                for prediction in predictions:

                    (
                        x1,
                        y1,
                        x2,
                        y2,
                        confidence,
                        class_id,
                    ) = prediction.tolist()


                    class_id = int(
                        class_id
                    )

                    confidence = float(
                        confidence
                    )


                    class_name = (
                        self._get_class_name(
                            class_id
                        )
                    )


                    x1 = int(round(x1))
                    y1 = int(round(y1))
                    x2 = int(round(x2))
                    y2 = int(round(y2))


                    detection_data = {

                        "class_id": class_id,

                        "class_name": (
                            class_name
                        ),

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


                    detections.append(
                        detection_data
                    )


                    label = (

                        f"{class_name} "

                        f"{confidence * 100:.1f}%"

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


                # Confidence tertinggi di urutan pertama
                detections.sort(

                    key=lambda item: (
                        item["confidence"]
                    ),

                    reverse=True,

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
                "YOLOv5 PyTorch CPU"
            ),

            "image_size": (
                self.image_size
            ),

            "classes": (
                self.class_names
            ),

        }
