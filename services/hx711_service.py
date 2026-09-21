from __future__ import annotations

from collections import deque
import logging
import statistics
import threading
import time


logger = logging.getLogger(__name__)


class HX711Service:
    """Read HX711 data without blocking Flask startup."""

    def __init__(
        self,
        dt_pin: int,
        sck_pin: int,
        calibration_factor: float,
        raw_samples: int = 5,
        tare_samples: int = 20,
        filter_size: int = 10,
        read_timeout: float = 3.0,
        reconnect_interval: float = 3.0,
        log_interval: float = 30.0,
    ) -> None:
        self.dt_pin = dt_pin
        self.sck_pin = sck_pin
        self.calibration_factor = calibration_factor
        self.raw_samples = raw_samples
        self.tare_samples = tare_samples
        self.filter_size = filter_size
        self.read_timeout = max(0.5, float(read_timeout))
        self.reconnect_interval = max(0.5, float(reconnect_interval))
        self.log_interval = max(5.0, float(log_interval))

        self._hx = None
        self._gpio = None
        self._tare_value = 0.0
        self._weight = 0.0
        self._connected = False
        self._initializing = False
        self._last_error: str | None = "Sensor HX711 belum diinisialisasi."
        self._last_success_at: float | None = None
        self._retry_count = 0
        self._timing_warning_count = 0
        self._last_logged_error: str | None = None
        self._last_log_at = 0.0

        self._buffer: deque[float] = deque(maxlen=filter_size)
        self._state_lock = threading.Lock()
        self._io_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._stop_event = threading.Event()

    def _set_state(
        self,
        *,
        connected: bool | None = None,
        initializing: bool | None = None,
        error: str | None = None,
    ) -> None:
        with self._state_lock:
            if connected is not None:
                self._connected = connected
            if initializing is not None:
                self._initializing = initializing
            if error is not None:
                self._last_error = error

    def _log_sensor_issue(self, error: Exception) -> None:
        message = str(error)
        now = time.monotonic()
        if (
            self._last_logged_error is None
            or now - self._last_log_at >= self.log_interval
        ):
            logger.warning(
                "HX711 belum siap: %s; mencoba lagi dalam %.1f detik.",
                message,
                self.reconnect_interval,
            )
            self._last_logged_error = message
            self._last_log_at = now

    def _set_channel_gain_safely(self, num: int) -> bool:
        """Set gain without the unbounded recursive retry in hx711 1.1.2.3."""
        if not 1 <= num <= 3:
            raise ValueError("Jumlah pulsa gain HX711 harus antara 1 dan 3.")

        for _ in range(num):
            started_at = time.perf_counter()
            self._gpio.output(self.sck_pin, True)
            self._gpio.output(self.sck_pin, False)
            if time.perf_counter() - started_at >= 0.00006:
                with self._state_lock:
                    self._timing_warning_count += 1
        return True

    def _initialize_hardware(self) -> None:
        import RPi.GPIO as GPIO
        from hx711 import HX711

        GPIO.setwarnings(False)
        self._gpio = GPIO
        self._hx = HX711(dout_pin=self.dt_pin, pd_sck_pin=self.sck_pin)

        # The installed driver recursively calls get_raw_data() whenever a
        # clock pulse exceeds 60 us. On Linux that can recurse indefinitely.
        self._hx._set_channel_gain = self._set_channel_gain_safely

    def _read_raw_mean(self, samples: int) -> float:
        if self._hx is None:
            raise RuntimeError("HX711 belum diinisialisasi.")

        deadline = time.monotonic() + self.read_timeout
        values: list[float] = []
        attempts = 0
        max_attempts = max(samples * 10, 20)

        while (
            len(values) < samples
            and attempts < max_attempts
            and time.monotonic() < deadline
            and not self._stop_event.is_set()
        ):
            attempts += 1
            value = self._hx._read()
            if value not in (False, -1, None) and not isinstance(value, bool):
                values.append(float(value))

        if len(values) < samples:
            raise TimeoutError(
                "Pembacaan HX711 timeout "
                f"({len(values)}/{samples} sampel valid dalam "
                f"{self.read_timeout:.1f} detik). Periksa VCC, GND, "
                "DOUT GPIO5, dan SCK GPIO6."
            )

        return statistics.mean(values)

    def _tare_unlocked(self) -> float:
        tare_value = self._read_raw_mean(self.tare_samples)
        with self._state_lock:
            self._tare_value = tare_value
            self._buffer.clear()
            self._weight = 0.0
        return tare_value

    def _connect_and_tare(self) -> None:
        self._set_state(
            connected=False,
            initializing=True,
            error="Menginisialisasi dan melakukan tare HX711.",
        )
        with self._io_lock:
            self._initialize_hardware()
            self._read_raw_mean(self.raw_samples)
            if self._stop_event.wait(0.3):
                return
            self._tare_unlocked()

        with self._state_lock:
            self._connected = True
            self._initializing = False
            self._last_error = None
            self._last_success_at = time.monotonic()
            self._retry_count = 0
            self._last_logged_error = None
        logger.info("HX711 terhubung dan tare awal berhasil.")

    def start(self) -> None:
        """Start the worker immediately; hardware setup stays in the worker."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._set_state(
            connected=False,
            initializing=True,
            error="Inisialisasi HX711 berjalan di background.",
        )
        self._thread = threading.Thread(
            target=self._read_loop,
            name="hx711-reader",
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self) -> None:
        while self._running and not self._stop_event.is_set():
            with self._state_lock:
                connected = self._connected

            if not connected:
                try:
                    self._connect_and_tare()
                except Exception as error:
                    with self._state_lock:
                        self._connected = False
                        self._initializing = False
                        self._last_error = str(error)
                        self._retry_count += 1
                    self._log_sensor_issue(error)
                    self._stop_event.wait(self.reconnect_interval)
                    continue

            try:
                with self._io_lock:
                    raw_value = self._read_raw_mean(self.raw_samples)

                with self._state_lock:
                    weight = (
                        raw_value - self._tare_value
                    ) / self.calibration_factor
                    if abs(weight) < 1:
                        weight = 0.0
                    self._buffer.append(weight)
                    self._weight = statistics.mean(self._buffer)
                    self._connected = True
                    self._last_error = None
                    self._last_success_at = time.monotonic()
            except Exception as error:
                with self._state_lock:
                    self._connected = False
                    self._last_error = str(error)
                    self._retry_count += 1
                self._log_sensor_issue(error)

            self._stop_event.wait(0.25)

    def tare(self) -> float:
        with self._state_lock:
            if not self._connected or self._hx is None:
                raise RuntimeError(
                    self._last_error or "Sensor HX711 belum tersedia."
                )

        if not self._io_lock.acquire(timeout=0.5):
            raise RuntimeError("Sensor HX711 sedang digunakan. Coba tare lagi.")
        try:
            return self._tare_unlocked()
        finally:
            self._io_lock.release()

    def get_weight(self) -> float:
        with self._state_lock:
            return round(float(self._weight), 2)

    def get_status(self) -> dict:
        with self._state_lock:
            age = (
                None
                if self._last_success_at is None
                else round(time.monotonic() - self._last_success_at, 1)
            )
            return {
                "connected": self._connected,
                "initializing": self._initializing,
                "error": self._last_error,
                "weight": round(float(self._weight), 2),
                "tare_value": round(self._tare_value, 2),
                "calibration_factor": self.calibration_factor,
                "filter": "moving_average",
                "filter_size": self.filter_size,
                "last_read_age_seconds": age,
                "retry_count": self._retry_count,
                "timing_warning_count": self._timing_warning_count,
            }

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.read_timeout + 1)

        if self._gpio is not None and not (
            self._thread and self._thread.is_alive()
        ):
            try:
                self._gpio.cleanup([self.dt_pin, self.sck_pin])
            except Exception:
                pass
