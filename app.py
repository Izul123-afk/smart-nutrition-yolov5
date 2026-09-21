from __future__ import annotations

import atexit
from datetime import datetime, timedelta, timezone
import logging
import os
import threading
import time
import traceback
import uuid

import cv2
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

import config
from services.camera_service import CameraService
from services.database_service import DatabaseService
from services.export_service import ExportService
from services.hx711_service import HX711Service
from services.nutrition_service import NutritionService
from services.yolo_service import YOLOService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
config.PHOTO_DIR.mkdir(parents=True, exist_ok=True)

database_service = DatabaseService(config.DATABASE_PATH, config.BASE_DIR)
nutrition_service = NutritionService(config.NUTRITION_CSV_PATH)
export_service = ExportService(database_service)
legacy_imported = database_service.import_legacy_images(config.RESULT_DIR)
if legacy_imported:
    logging.getLogger(__name__).info(
        "%s foto hasil lama dimasukkan ke database sebagai data legacy.",
        legacy_imported,
    )

camera_service = CameraService(
    camera_index=config.CAMERA_INDEX,
    width=config.CAMERA_WIDTH,
    height=config.CAMERA_HEIGHT,
    fps=config.CAMERA_FPS,
    jpeg_quality=config.CAMERA_JPEG_QUALITY,
)

hx711_service = HX711Service(
    dt_pin=config.HX711_DT_PIN,
    sck_pin=config.HX711_SCK_PIN,
    calibration_factor=config.HX711_CALIBRATION_FACTOR,
    raw_samples=config.HX711_RAW_SAMPLES,
    tare_samples=config.HX711_TARE_SAMPLES,
    filter_size=config.HX711_FILTER_SIZE,
    read_timeout=config.HX711_READ_TIMEOUT_SECONDS,
    reconnect_interval=config.HX711_RECONNECT_INTERVAL_SECONDS,
    log_interval=config.HX711_LOG_INTERVAL_SECONDS,
)

yolo_service: YOLOService | None = None
yolo_startup_error: str | None = None

try:
    yolo_service = YOLOService(
        model_path=config.MODEL_PATH,
        yolov5_path=config.YOLOV5_REPO,
        confidence=config.YOLO_CONFIDENCE,
        iou_threshold=config.YOLO_IOU_THRESHOLD,
        image_size=config.YOLO_IMAGE_SIZE,
        max_detections=config.YOLO_MAX_DETECTIONS,
        torch_threads=config.TORCH_THREADS,
    )
except Exception as error:
    yolo_startup_error = str(error)
    traceback.print_exc()
    print("Model YOLOv5 gagal dimuat:", error)

camera_service.start()
hx711_service.start()

scan_lock = threading.Lock()

# Preset filter waktu riwayat. None berarti tanpa batas (semua data).
HISTORY_RANGES: dict[str, timedelta | None] = {
    "6h": timedelta(hours=6),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "all": None,
}


def _parse_history_range(value: str | None) -> tuple[str, datetime | None]:
    """Kembalikan (nama_range, batas_awal_utc). ValueError bila tidak dikenal."""
    key = (value or "all").strip().lower()
    if key not in HISTORY_RANGES:
        raise ValueError(
            "Parameter range harus salah satu dari: "
            + ", ".join(HISTORY_RANGES)
        )
    delta = HISTORY_RANGES[key]
    since = datetime.now(timezone.utc) - delta if delta else None
    return key, since


def _model_status() -> dict:
    if yolo_service is None:
        return {
            "loaded": False,
            "error": yolo_startup_error,
            "model": config.MODEL_PATH.name,
            "backend": "YOLOv5 PyTorch CPU",
            "image_size": config.YOLO_IMAGE_SIZE,
            "input_width": config.YOLO_IMAGE_SIZE,
            "input_height": config.YOLO_IMAGE_SIZE,
        }

    status = dict(yolo_service.get_status())
    image_size = int(status.get("image_size", config.YOLO_IMAGE_SIZE))
    status.setdefault("input_width", image_size)
    status.setdefault("input_height", image_size)
    return status


def _success_flag(data: dict) -> dict:
    """Tambahkan alias respons agar frontend lama dan baru tetap kompatibel."""
    data["success"] = True
    data["ok"] = True
    return data


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    return Response(
        camera_service.mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
def api_status():
    camera_status = camera_service.get_status()
    sensor_status = hx711_service.get_status()
    model_status = _model_status()

    return jsonify(
        _success_flag(
            {
                "camera": camera_status,
                # Dua nama disediakan untuk kompatibilitas.
                "sensor": sensor_status,
                "scale": sensor_status,
                "model": model_status,
                "database": database_service.status(),
                "camera_connected": bool(camera_status.get("connected")),
                "weight": float(sensor_status.get("weight", 0.0)),
                "weight_grams": float(sensor_status.get("weight", 0.0)),
                "model_loaded": bool(model_status.get("loaded")),
            }
        )
    )


@app.get("/api/weight")
def api_weight():
    status = hx711_service.get_status()
    weight = float(status.get("weight", 0.0))

    return jsonify(
        _success_flag(
            {
                "connected": bool(status.get("connected")),
                "error": status.get("error"),
                "weight": weight,
                "weight_grams": weight,
            }
        )
    )


@app.post("/api/tare")
def api_tare():
    try:
        tare_value = float(hx711_service.tare())
        weight = float(hx711_service.get_weight())
        return jsonify(
            _success_flag(
                {
                    "message": "Tare berhasil.",
                    "tare_value": round(tare_value, 2),
                    "tare_raw": round(tare_value, 2),
                    "weight": round(weight, 2),
                    "weight_grams": round(weight, 2),
                }
            )
        )
    except Exception as error:
        traceback.print_exc()
        return jsonify(
            {
                "success": False,
                "ok": False,
                "error": str(error),
            }
        ), 500


def _write_jpeg_atomic(output_path, frame) -> None:
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


def _remove_incomplete_scan(scan_directory, *paths) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        scan_directory.rmdir()
    except OSError:
        pass


def _add_image_urls(record: dict) -> dict:
    scan_id = record.get("scan_id")
    if not scan_id:
        return record
    if record.get("annotated_image_path"):
        record["image_url"] = url_for(
            "api_scan_image",
            scan_id=scan_id,
            image_kind="annotated",
        )
        record["result_image"] = record["image_url"]
        record["image"] = record["image_url"]
    if record.get("original_image_path"):
        record["original_image_url"] = url_for(
            "api_scan_image",
            scan_id=scan_id,
            image_kind="original",
        )
    return record


def run_scan(frame, source: str, capture_ms: float = 0.0) -> dict:
    if yolo_service is None:
        raise RuntimeError(yolo_startup_error or "Model YOLOv5 belum tersedia.")

    complete_started = time.perf_counter()
    original_frame = frame.copy()
    result = yolo_service.predict(frame)
    annotated_frame = result.pop("annotated_frame")

    scan_id = uuid.uuid4().hex[:12]
    captured_at = datetime.now(timezone.utc)
    timestamp = captured_at.isoformat(timespec="seconds")
    scan_directory = (
        config.PHOTO_DIR
        / captured_at.strftime("%Y")
        / captured_at.strftime("%m")
        / captured_at.strftime("%d")
        / scan_id
    )
    original_path = scan_directory / "original.jpg"
    annotated_path = scan_directory / "annotated.jpg"

    postprocess_started = time.perf_counter()
    try:
        _write_jpeg_atomic(original_path, original_frame)
        _write_jpeg_atomic(annotated_path, annotated_frame)
    except Exception:
        _remove_incomplete_scan(
            scan_directory,
            original_path,
            annotated_path,
        )
        raise
    image_save_ms = (time.perf_counter() - postprocess_started) * 1000

    weight_started = time.perf_counter()
    weight = float(hx711_service.get_weight())
    weight_ms = (time.perf_counter() - weight_started) * 1000

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
        "weight_ms": round(weight_ms, 2),
        "complete_ms": round(complete_ms, 2),
    }

    food_name = result.get("food_name", "Tidak terdeteksi")
    detections = result.get("detections", [])
    nutrition = nutrition_service.calculate(food_name, weight)
    model_status = _model_status()
    original_relative = original_path.relative_to(config.BASE_DIR).as_posix()
    annotated_relative = annotated_path.relative_to(config.BASE_DIR).as_posix()

    response_data = _success_flag(
        {
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
    )
    try:
        database_service.save_scan(
            response_data,
            detections,
            timings,
            nutrition,
        )
    except Exception:
        _remove_incomplete_scan(
            scan_directory,
            original_path,
            annotated_path,
        )
        raise
    _add_image_urls(response_data)
    return response_data


@app.post("/api/scan")
def api_scan():
    if not scan_lock.acquire(blocking=False):
        return jsonify(
            {
                "success": False,
                "ok": False,
                "error": "Proses scan lain masih berjalan.",
            }
        ), 409

    try:
        capture_started = time.perf_counter()
        frame = camera_service.get_latest_frame()
        capture_ms = (time.perf_counter() - capture_started) * 1000

        if frame is None:
            return jsonify(
                {
                    "success": False,
                    "ok": False,
                    "error": "Frame kamera belum tersedia.",
                }
            ), 503

        return jsonify(run_scan(frame, source="camera", capture_ms=capture_ms))
    except Exception as error:
        traceback.print_exc()
        return jsonify(
            {
                "success": False,
                "ok": False,
                "error": str(error),
            }
        ), 500
    finally:
        scan_lock.release()


@app.get("/api/test-image")
def api_test_image():
    if not scan_lock.acquire(blocking=False):
        return jsonify(
            {
                "success": False,
                "ok": False,
                "error": "Proses scan lain masih berjalan.",
            }
        ), 409

    try:
        capture_started = time.perf_counter()
        frame = cv2.imread(str(config.TEST_IMAGE_PATH))
        capture_ms = (time.perf_counter() - capture_started) * 1000

        if frame is None:
            return jsonify(
                {
                    "success": False,
                    "ok": False,
                    "error": f"Gambar uji gagal dibaca: {config.TEST_IMAGE_PATH}",
                }
            ), 400

        return jsonify(run_scan(frame, source="test_image", capture_ms=capture_ms))
    except Exception as error:
        traceback.print_exc()
        return jsonify(
            {
                "success": False,
                "ok": False,
                "error": str(error),
            }
        ), 500
    finally:
        scan_lock.release()


@app.get("/api/history")
def api_history():
    try:
        limit = min(500, max(1, int(request.args.get("limit", 100))))
        offset = max(0, int(request.args.get("offset", 0)))
    except ValueError:
        return jsonify({"success": False, "ok": False, "error": "Parameter pagination tidak valid."}), 400
    try:
        range_key, since = _parse_history_range(request.args.get("range"))
    except ValueError as error:
        return jsonify({"success": False, "ok": False, "error": str(error)}), 400

    records = database_service.list_scans(
        limit=limit, offset=offset, since=since
    )
    for record in records:
        _add_image_urls(record)

    return jsonify(
        {
            "success": True,
            "ok": True,
            "range": range_key,
            "since": since.isoformat(timespec="seconds") if since else None,
            "count": database_service.count_scans(since=since),
            "total_count": database_service.count_scans(),
            "limit": limit,
            "offset": offset,
            "records": records,
        }
    )


@app.get("/api/scans/<scan_id>")
def api_scan_detail(scan_id: str):
    record = database_service.get_scan(scan_id)
    if record is None:
        return jsonify({"success": False, "ok": False, "error": "Data scan tidak ditemukan."}), 404
    return jsonify(_success_flag(_add_image_urls(record)))


@app.get("/api/scans/<scan_id>/image/<image_kind>")
def api_scan_image(scan_id: str, image_kind: str):
    if image_kind not in {"original", "annotated"}:
        abort(404)
    record = database_service.get_scan(scan_id)
    if record is None:
        abort(404)

    relative_path = record.get(f"{image_kind}_image_path")
    if not relative_path:
        abort(404)
    image_path = (config.BASE_DIR / relative_path).resolve()
    try:
        image_path.relative_to(config.BASE_DIR.resolve())
    except ValueError:
        abort(404)
    if not image_path.is_file():
        abort(404)
    return send_file(image_path, mimetype="image/jpeg", conditional=True)


@app.post("/api/history/clear")
def api_history_clear():
    deleted_count = database_service.soft_delete_all()

    return jsonify(
        {
            "success": True,
            "ok": True,
            "deleted_count": deleted_count,
            "message": "Riwayat disembunyikan. File foto tetap tersimpan.",
        }
    )


@app.get("/api/export/<export_format>")
def api_export(export_format: str):
    try:
        _range_key, since = _parse_history_range(request.args.get("range"))
        artifact = export_service.create(export_format, since=since)
    except ValueError as error:
        return jsonify({"success": False, "ok": False, "error": str(error)}), 400
    except ImportError as error:
        return jsonify({"success": False, "ok": False, "error": f"Dependency export belum tersedia: {error}"}), 503

    return send_file(
        artifact.content,
        mimetype=artifact.mimetype,
        as_attachment=True,
        download_name=artifact.filename,
        max_age=0,
    )


@app.errorhandler(404)
def not_found(_error):
    return jsonify(
        {
            "success": False,
            "ok": False,
            "error": "Endpoint tidak ditemukan.",
        }
    ), 404


def shutdown_services() -> None:
    camera_service.stop()
    hx711_service.stop()


atexit.register(shutdown_services)


if __name__ == "__main__":
    print(f"Dashboard berjalan di http://127.0.0.1:{config.PORT}")
    print(f"Akses melalui jaringan: http://IP_RASPBERRY:{config.PORT}")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
