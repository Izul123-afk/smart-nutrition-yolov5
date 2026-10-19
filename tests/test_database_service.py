from pathlib import Path
import importlib.util
import tempfile
import unittest

from services.database_service import DatabaseService
from services.export_service import ExportService
from services.nutrition_service import NutritionService


class DatabaseServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temporary_directory.name)
        self.database = DatabaseService(
            self.base_dir / "data" / "food_recognition.db",
            self.base_dir,
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    @staticmethod
    def scan_record():
        return {
            "scan_id": "abc123",
            "timestamp": "2026-09-13T03:00:00+00:00",
            "source": "camera",
            "detected": True,
            "food_name": "Rawon",
            "confidence_percent": 91.5,
            "confidence_ratio": 0.915,
            "weight_grams": 125.0,
            "object_count": 1,
            "model_name": "best2.pt",
            "model_input": 640,
            "original_image_path": "data/photos/abc123/original.jpg",
            "annotated_image_path": "data/photos/abc123/annotated.jpg",
        }

    @staticmethod
    def timings():
        return {
            "capture_ms": 1,
            "preprocess_ms": 2,
            "inference_ms": 300,
            "postprocess_ms": 4,
            "weight_ms": 1,
            "complete_ms": 308,
        }

    @staticmethod
    def nutrition():
        return {
            "available": True,
            "data_version": "test123",
            "calculated_weight_grams": 125,
            "per_100g": {
                "energy_kcal": 60,
                "protein_g": 5.4,
                "fat_g": 2.5,
                "carbohydrate_g": 4,
            },
            "totals": {
                "energy_kcal": 75,
                "protein_g": 6.75,
                "fat_g": 3.125,
                "carbohydrate_g": 5,
            },
        }

    def test_scan_and_children_are_persistent(self):
        detections = [{
            "class_id": 4,
            "class_name": "Rawon",
            "confidence": 91.5,
            "box": {"x1": 10, "y1": 20, "x2": 100, "y2": 120},
        }]
        self.database.save_scan(
            self.scan_record(),
            detections,
            self.timings(),
            self.nutrition(),
        )

        reopened = DatabaseService(self.database.database_path, self.base_dir)
        record = reopened.get_scan("abc123")

        self.assertEqual(reopened.count_scans(), 1)
        self.assertEqual(record["food_name"], "Rawon")
        self.assertEqual(record["timings"]["inference_ms"], 300)
        self.assertEqual(record["nutrition"]["totals"]["energy_kcal"], 75)
        self.assertEqual(record["detections"][0]["class_id"], 4)

    def test_record_exposes_image_filenames(self):
        self.database.save_scan(
            self.scan_record(), [], self.timings(), self.nutrition()
        )
        record = self.database.list_scans()[0]

        self.assertEqual(record["original_filename"], "original.jpg")
        self.assertEqual(record["annotated_filename"], "annotated.jpg")
        self.assertEqual(
            record["annotated_image_path"], "data/photos/abc123/annotated.jpg"
        )

    def test_since_filters_history_count_and_export(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        for scan_id, age in (("new1", 1), ("old1", 48), ("older", 24 * 10)):
            record = self.scan_record()
            record["scan_id"] = scan_id
            record["timestamp"] = (now - timedelta(hours=age)).isoformat(
                timespec="seconds"
            )
            self.database.save_scan(record, [], self.timings(), self.nutrition())

        since_24h = now - timedelta(hours=24)
        since_7d = now - timedelta(days=7)

        self.assertEqual(self.database.count_scans(), 3)
        self.assertEqual(self.database.count_scans(since=since_24h), 1)
        self.assertEqual(self.database.count_scans(since=since_7d), 2)
        self.assertEqual(
            [r["scan_id"] for r in self.database.list_scans(since=since_7d)],
            ["new1", "old1"],
        )
        self.assertEqual(
            len(self.database.export_dataset(since=since_24h)["scans"]), 1
        )
        self.assertEqual(len(self.database.list_scans(since=None)), 3)

    def test_clear_is_soft_delete_and_keeps_database_record(self):
        self.database.save_scan(
            self.scan_record(), [], self.timings(), self.nutrition()
        )

        self.assertEqual(self.database.soft_delete_all(), 1)
        self.assertEqual(self.database.count_scans(), 0)
        self.assertIsNone(self.database.get_scan("abc123"))

        with self.database._reader() as connection:
            count = connection.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        self.assertEqual(count, 1)

    def test_csv_export_contains_scan_and_detection_json(self):
        self.database.save_scan(
            self.scan_record(), [], self.timings(), self.nutrition()
        )
        artifact = ExportService(self.database).csv()
        content = artifact.content.getvalue().decode("utf-8-sig")

        self.assertIn("food_name", content)
        self.assertIn("Rawon", content)
        self.assertIn("detections_json", content)

    @unittest.skipUnless(
        importlib.util.find_spec("openpyxl")
        and importlib.util.find_spec("reportlab"),
        "Dependency XLSX/PDF tidak tersedia di environment ini.",
    )
    def test_xlsx_and_pdf_exports_have_valid_signatures(self):
        self.database.save_scan(
            self.scan_record(), [], self.timings(), self.nutrition()
        )
        exporter = ExportService(self.database)

        self.assertTrue(exporter.xlsx().content.getvalue().startswith(b"PK"))
        self.assertTrue(exporter.pdf().content.getvalue().startswith(b"%PDF"))


class NutritionServiceTests(unittest.TestCase):
    def test_calculation_uses_csv_and_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "nutrition.csv"
            csv_path.write_text(
                "No,Kelas_Makanan,Energi_kkal_100g,Protein_g_100g,"
                "Lemak_g_100g,Karbohidrat_g_100g\n"
                "1,Rawon,60,5.4,2.5,4\n",
                encoding="utf-8",
            )
            nutrition = NutritionService(csv_path).calculate("Rawon", 125)

        self.assertTrue(nutrition["available"])
        self.assertEqual(nutrition["totals"]["energy_kcal"], 75)
        self.assertEqual(nutrition["totals"]["protein_g"], 6.75)


if __name__ == "__main__":
    unittest.main()
