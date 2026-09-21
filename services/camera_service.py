from __future__ import annotations

import threading
import time
from typing import Generator

import cv2
import numpy as np


class CameraService:
    """
    Service untuk mengambil frame kamera secara kontinu.

    Kamera hanya digunakan untuk preview.
    YOLO dijalankan ketika tombol scan ditekan.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        jpeg_quality: int = 80,
    ) -> None:

        self.camera_index = camera_index

        self.width = width
        self.height = height

        self.fps = fps

        self.jpeg_quality = jpeg_quality

        self._capture: cv2.VideoCapture | None = None

        self._frame: np.ndarray | None = None

        self._frame_lock = threading.Lock()

        self._thread: threading.Thread | None = None

        self._running = False

        self._connected = False

        self._last_error: str | None = None

        self._frame_counter = 0

        self._measured_fps = 0.0


    # =====================================================
    # MEMBUKA KAMERA
    # =====================================================

    def _open_camera(self) -> bool:

        self._release_camera()

        import platform

        backends = []
        system = platform.system()
        if system == "Darwin":
            backends = [cv2.CAP_AVFOUNDATION, None]
        elif system == "Linux":
            backends = [cv2.CAP_V4L2, None]
        else:
            backends = [None]

        capture = None
        for backend in backends:
            if backend is None:
                capture = cv2.VideoCapture(self.camera_index)
            else:
                capture = cv2.VideoCapture(self.camera_index, backend)
            if capture.isOpened():
                break
            capture.release()

        if capture is None or not capture.isOpened():

            self._connected = False

            self._last_error = (
                f"Kamera index {self.camera_index} "
                "tidak dapat dibuka."
            )

            return False


        # MJPG biasanya lebih ringan untuk webcam USB
        capture.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG"),
        )

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        capture.set(
            cv2.CAP_PROP_FPS,
            self.fps,
        )

        capture.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1,
        )


        self._capture = capture

        self._connected = True

        self._last_error = None

        return True


    # =====================================================
    # MEMULAI THREAD KAMERA
    # =====================================================

    def start(self) -> None:

        if self._running:
            return

        self._running = True

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
            daemon=True,
        )

        self._thread.start()


    # =====================================================
    # LOOP PEMBACAAN KAMERA
    # =====================================================

    def _capture_loop(self) -> None:

        fps_started_at = time.perf_counter()

        fps_frame_count = 0


        while self._running:

            if (
                self._capture is None
                or not self._capture.isOpened()
            ):

                if not self._open_camera():

                    time.sleep(1)

                    continue


            success, frame = self._capture.read()


            if not success or frame is None:

                self._connected = False

                self._last_error = (
                    "Frame kamera gagal dibaca."
                )

                self._release_camera()

                time.sleep(0.5)

                continue


            with self._frame_lock:

                self._frame = frame

                self._frame_counter += 1


            self._connected = True

            self._last_error = None


            # Hitung FPS aktual
            fps_frame_count += 1

            elapsed = (
                time.perf_counter()
                - fps_started_at
            )

            if elapsed >= 1:

                self._measured_fps = (
                    fps_frame_count / elapsed
                )

                fps_frame_count = 0

                fps_started_at = (
                    time.perf_counter()
                )


    # =====================================================
    # MENGAMBIL FRAME TERBARU
    # =====================================================

    def get_latest_frame(
        self,
    ) -> np.ndarray | None:

        with self._frame_lock:

            if self._frame is None:
                return None

            return self._frame.copy()


    # =====================================================
    # STREAM MJPEG UNTUK DASHBOARD
    # =====================================================

    def mjpeg_generator(
        self,
    ) -> Generator[bytes, None, None]:

        encode_parameters = [

            cv2.IMWRITE_JPEG_QUALITY,

            self.jpeg_quality,

        ]


        while self._running:

            frame = self.get_latest_frame()


            if frame is None:

                time.sleep(0.05)

                continue


            success, encoded = cv2.imencode(

                ".jpg",

                frame,

                encode_parameters,

            )


            if not success:

                time.sleep(0.02)

                continue


            yield (

                b"--frame\r\n"

                b"Content-Type: image/jpeg\r\n\r\n"

                + encoded.tobytes()

                + b"\r\n"

            )


    # =====================================================
    # STATUS KAMERA
    # =====================================================

    def get_status(self) -> dict:

        frame = self.get_latest_frame()


        if frame is not None:

            height, width = frame.shape[:2]

        else:

            width = 0

            height = 0


        return {

            "connected": self._connected,

            "error": self._last_error,

            "width": width,

            "height": height,

            "fps": round(
                self._measured_fps,
                1,
            ),

            "frame_count": (
                self._frame_counter
            ),

        }


    # =====================================================
    # MELEPASKAN KAMERA
    # =====================================================

    def _release_camera(self) -> None:

        if self._capture is not None:

            self._capture.release()

            self._capture = None


    # =====================================================
    # MENGHENTIKAN KAMERA
    # =====================================================

    def stop(self) -> None:

        self._running = False


        if (
            self._thread
            and self._thread.is_alive()
        ):

            self._thread.join(
                timeout=2
            )


        self._release_camera()