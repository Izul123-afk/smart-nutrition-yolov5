import threading
import time
import unittest

from services.hx711_service import HX711Service


class FakeHX711:
    def __init__(self, values):
        self.values = iter(values)

    def _read(self):
        return next(self.values, False)


class HX711ServiceTests(unittest.TestCase):
    def make_service(self, **overrides):
        options = {
            "dt_pin": 5,
            "sck_pin": 6,
            "calibration_factor": 100.0,
            "raw_samples": 2,
            "tare_samples": 2,
            "filter_size": 3,
            "read_timeout": 0.01,
            "reconnect_interval": 0.01,
            "log_interval": 5.0,
        }
        options.update(overrides)
        return HX711Service(**options)

    def test_valid_samples_are_averaged(self):
        service = self.make_service()
        service._hx = FakeHX711([False, 10, 20])

        self.assertEqual(service._read_raw_mean(2), 15.0)

    def test_invalid_samples_have_bounded_attempts(self):
        service = self.make_service()
        service._hx = FakeHX711([False] * 100)
        started_at = time.monotonic()

        with self.assertRaises(TimeoutError):
            service._read_raw_mean(2)

        self.assertLess(time.monotonic() - started_at, 0.2)

    def test_start_does_not_wait_for_hardware_initialization(self):
        service = self.make_service()
        release = threading.Event()

        def slow_initialization():
            release.wait(1)
            raise RuntimeError("sensor unavailable")

        service._connect_and_tare = slow_initialization
        started_at = time.monotonic()
        service.start()
        elapsed = time.monotonic() - started_at
        release.set()
        service.stop()

        self.assertLess(elapsed, 0.2)


if __name__ == "__main__":
    unittest.main()
