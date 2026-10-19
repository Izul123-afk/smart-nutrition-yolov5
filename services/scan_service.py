from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import time
import uuid

import cv2


class ScanService:
    """Menyimpan satu hasil scan: foto, nutrisi, dan baris database.

    Dipakai bersama oleh scan web (``app.py``) dan scan terminal
    (``manual_scan.py``) supaya keduanya menulis record yang identik
    ke database yang sama.
    """

    def __init__(
        self,
        yolo_service,
        nutrition_service,
        database_service,
        base_dir: str | Path,
        photo_dir: str | Path,
    ) -> None:
        self.yolo_service = yolo_service
        self.nutrition_service = nutrition_service
        self.database_service = database_service
        self.base_dir = Path(base_dir).resolve()
        self.photo_dir = Path(photo_dir).resolve()

    @staticmethod
    def _write_jpeg_atomic(output_path: Path, frame) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f".{output_path.stem}.{uuid.uuid4().hex}.tmp.jpg"
        )
        try:
            if not cv2.imwrite(str(temporary_path), frame):
                raise RuntimeError(f"Gagal menulis gambar: {output_path}")
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _remove_incomplete_scan(scan_directory: Path, *paths: Path) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            scan_directory.rmdir()
        except OSError:
            pass

    def record(
        self,
        frame,
        source: str,
        weight_grams: float,
        capture_ms: float = 0.0,
    ) -> dict:
        """Deteksi, simpan foto, hitung nutrisi, lalu tulis ke database."""
        complete_started = time.perf_counter()
        original_frame = frame.copy()
        result = self.yolo_service.predict(frame)
        annotated_frame = result.pop("annotated_frame")

        scan_id = uuid.uuid4().hex[:12]
        captured_at = datetime.now(timezone.utc)
        timestamp = captured_at.isoformat(timespec="seconds")
        scan_directory = (
            self.photo_dir
            / captured_at.strftime("%Y")
            / captured_at.strftime("%m")
            / captured_at.strftime("%d")
            / scan_id
        )
        original_path = scan_directory / "original.jpg"
        annotated_path = scan_directory / "annotated.jpg"

        postprocess_started = time.perf_counter()
        try:
            self._write_jpeg_atomic(original_path, original_frame)
            self._write_jpeg_atomic(annotated_path, annotated_frame)
        except Exception:
            self._remove_incomplete_scan(
                scan_directory,
                original_path,
                annotated_path,
            )
            raise
        image_save_ms = (time.perf_counter() - postprocess_started) * 1000

        weight = float(weight_grams)

        confidence_percent = float(result.get("confidence", 0.0))
        confidence_ratio = (
            confidence_percent / 100.0
            if confidence_percent > 1.0
            else confidence_percent
        )

        service_timings = result.get("timings") or {}
        inference_ms = float(
            service_timings.get("inference_ms", result.get("inference_ms", 0.0))
        )
        preprocess_ms = float(service_timings.get("preprocess_ms", 0.0))
        model_postprocess_ms = float(service_timings.get("postprocess_ms", 0.0))
        postprocess_ms = model_postprocess_ms + image_save_ms
        complete_ms = (time.perf_counter() - complete_started) * 1000 + capture_ms

        timings = {
            "capture_ms": round(float(capture_ms), 2),
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "postprocess_ms": round(postprocess_ms, 2),
            "weight_ms": 0.0,
            "complete_ms": round(complete_ms, 2),
        }

        food_name = result.get("food_name", "Tidak terdeteksi")
        detections = result.get("detections", [])
        nutrition = self.nutrition_service.calculate(food_name, weight)
        model_status = self.yolo_service.get_status()
        original_relative = original_path.relative_to(self.base_dir).as_posix()
        annotated_relative = annotated_path.relative_to(self.base_dir).as_posix()

        response_data = {
            "scan_id": scan_id,
            "timestamp": timestamp,
            "source": source,
            "detected": bool(result.get("detected")),
            "food_name": food_name,
            "makanan": food_name,
            "label": food_name,
            # Dipertahankan dalam persen seperti service saat ini.
            "confidence": round(confidence_percent, 2),
            "confidence_percent": round(confidence_percent, 2),
            "confidence_ratio": round(confidence_ratio, 6),
            "weight": round(weight, 2),
            "berat": round(weight, 2),
            "weight_grams": round(weight, 2),
            "object_count": int(result.get("object_count", 0)),
            "jumlah_objek": int(result.get("object_count", 0)),
            "inference_ms": round(inference_ms, 2),
            "inference_time_ms": round(inference_ms, 2),
            "total_ms": round(complete_ms, 2),
            "timings": timings,
            "detections": detections,
            "nutrition": nutrition,
            "model_name": model_status.get("model"),
            "model_input": model_status.get("image_size"),
            "original_image_path": original_relative,
            "annotated_image_path": annotated_relative,
        }
        response_data["success"] = True
        response_data["ok"] = True

        try:
            self.database_service.save_scan(
                response_data,
                detections,
                timings,
                nutrition,
            )
        except Exception:
            self._remove_incomplete_scan(
                scan_directory,
                original_path,
                annotated_path,
            )
            raise
        return response_data
