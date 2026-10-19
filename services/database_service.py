from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Iterator


class DatabaseService:
    """Persistent local storage for scan metadata and result file paths."""

    SCHEMA_VERSION = 1

    def __init__(self, database_path: str | Path, base_dir: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.base_dir = Path(base_dir).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        finally:
            connection.close()

        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY,
                    captured_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    detected INTEGER NOT NULL DEFAULT 0,
                    food_name TEXT NOT NULL,
                    confidence_percent REAL NOT NULL DEFAULT 0,
                    confidence_ratio REAL NOT NULL DEFAULT 0,
                    weight_grams REAL NOT NULL DEFAULT 0,
                    object_count INTEGER NOT NULL DEFAULT 0,
                    model_name TEXT,
                    model_input INTEGER,
                    original_image_path TEXT,
                    annotated_image_path TEXT,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id TEXT NOT NULL REFERENCES scans(scan_id)
                        ON DELETE CASCADE,
                    class_id INTEGER NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence_percent REAL NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scan_timings (
                    scan_id TEXT PRIMARY KEY REFERENCES scans(scan_id)
                        ON DELETE CASCADE,
                    capture_ms REAL NOT NULL DEFAULT 0,
                    preprocess_ms REAL NOT NULL DEFAULT 0,
                    inference_ms REAL NOT NULL DEFAULT 0,
                    postprocess_ms REAL NOT NULL DEFAULT 0,
                    weight_ms REAL NOT NULL DEFAULT 0,
                    complete_ms REAL NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS nutrition_snapshots (
                    scan_id TEXT PRIMARY KEY REFERENCES scans(scan_id)
                        ON DELETE CASCADE,
                    available INTEGER NOT NULL DEFAULT 0,
                    data_version TEXT,
                    energy_kcal_per_100g REAL,
                    protein_g_per_100g REAL,
                    fat_g_per_100g REAL,
                    carbohydrate_g_per_100g REAL,
                    calculated_weight_grams REAL NOT NULL DEFAULT 0,
                    energy_kcal_total REAL,
                    protein_g_total REAL,
                    fat_g_total REAL,
                    carbohydrate_g_total REAL
                );

                CREATE INDEX IF NOT EXISTS idx_scans_captured_at
                    ON scans(captured_at DESC);
                CREATE INDEX IF NOT EXISTS idx_scans_food_name
                    ON scans(food_name);
                CREATE INDEX IF NOT EXISTS idx_scans_deleted_at
                    ON scans(deleted_at);
                CREATE INDEX IF NOT EXISTS idx_detections_scan_id
                    ON detections(scan_id);
                """
            )
            connection.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def save_scan(
        self,
        scan: dict,
        detections: list[dict],
        timings: dict,
        nutrition: dict,
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO scans (
                    scan_id, captured_at, source, detected, food_name,
                    confidence_percent, confidence_ratio, weight_grams,
                    object_count, model_name, model_input,
                    original_image_path, annotated_image_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan["scan_id"],
                    scan["timestamp"],
                    scan.get("source", "camera"),
                    int(bool(scan.get("detected"))),
                    scan.get("food_name", "Tidak terdeteksi"),
                    float(scan.get("confidence_percent", 0)),
                    float(scan.get("confidence_ratio", 0)),
                    float(scan.get("weight_grams", 0)),
                    int(scan.get("object_count", 0)),
                    scan.get("model_name"),
                    scan.get("model_input"),
                    scan.get("original_image_path"),
                    scan.get("annotated_image_path"),
                    self._now(),
                ),
            )

            for detection in detections:
                box = detection.get("box") or {}
                connection.execute(
                    """
                    INSERT INTO detections (
                        scan_id, class_id, class_name, confidence_percent,
                        x1, y1, x2, y2
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        scan["scan_id"],
                        int(detection.get("class_id", -1)),
                        str(detection.get("class_name", "Tidak diketahui")),
                        float(detection.get("confidence", 0)),
                        int(box.get("x1", 0)),
                        int(box.get("y1", 0)),
                        int(box.get("x2", 0)),
                        int(box.get("y2", 0)),
                    ),
                )

            connection.execute(
                """
                INSERT INTO scan_timings (
                    scan_id, capture_ms, preprocess_ms, inference_ms,
                    postprocess_ms, weight_ms, complete_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan["scan_id"],
                    float(timings.get("capture_ms", 0)),
                    float(timings.get("preprocess_ms", 0)),
                    float(timings.get("inference_ms", 0)),
                    float(timings.get("postprocess_ms", 0)),
                    float(timings.get("weight_ms", 0)),
                    float(timings.get("complete_ms", 0)),
                ),
            )

            per_100g = nutrition.get("per_100g") or {}
            totals = nutrition.get("totals") or {}
            connection.execute(
                """
                INSERT INTO nutrition_snapshots (
                    scan_id, available, data_version,
                    energy_kcal_per_100g, protein_g_per_100g,
                    fat_g_per_100g, carbohydrate_g_per_100g,
                    calculated_weight_grams, energy_kcal_total,
                    protein_g_total, fat_g_total, carbohydrate_g_total
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan["scan_id"],
                    int(bool(nutrition.get("available"))),
                    nutrition.get("data_version"),
                    per_100g.get("energy_kcal"),
                    per_100g.get("protein_g"),
                    per_100g.get("fat_g"),
                    per_100g.get("carbohydrate_g"),
                    float(nutrition.get("calculated_weight_grams", 0)),
                    totals.get("energy_kcal"),
                    totals.get("protein_g"),
                    totals.get("fat_g"),
                    totals.get("carbohydrate_g"),
                ),
            )

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["detected"] = bool(record["detected"])
        record["timestamp"] = record.pop("captured_at")
        record["label"] = record["food_name"]
        for kind in ("original", "annotated"):
            path = record.get(f"{kind}_image_path")
            record[f"{kind}_filename"] = Path(path).name if path else None
        record["confidence"] = record["confidence_percent"]
        record["weight"] = record["weight_grams"]
        record["total_ms"] = record.get("complete_ms", 0)
        record["timings"] = {
            key: record.get(key, 0)
            for key in (
                "capture_ms",
                "preprocess_ms",
                "inference_ms",
                "postprocess_ms",
                "weight_ms",
                "complete_ms",
            )
        }
        record["nutrition"] = {
            "available": bool(record.get("nutrition_available", 0)),
            "data_version": record.get("data_version"),
            "calculated_weight_grams": record.get(
                "calculated_weight_grams", 0
            ),
            "per_100g": {
                "energy_kcal": record.get("energy_kcal_per_100g"),
                "protein_g": record.get("protein_g_per_100g"),
                "fat_g": record.get("fat_g_per_100g"),
                "carbohydrate_g": record.get("carbohydrate_g_per_100g"),
            },
            "totals": {
                "energy_kcal": record.get("energy_kcal_total"),
                "protein_g": record.get("protein_g_total"),
                "fat_g": record.get("fat_g_total"),
                "carbohydrate_g": record.get("carbohydrate_g_total"),
            },
        }
        return record

    @staticmethod
    def _select_sql() -> str:
        return """
            SELECT
                s.*,
                t.capture_ms, t.preprocess_ms, t.inference_ms,
                t.postprocess_ms, t.weight_ms, t.complete_ms,
                n.available AS nutrition_available, n.data_version,
                n.energy_kcal_per_100g, n.protein_g_per_100g,
                n.fat_g_per_100g, n.carbohydrate_g_per_100g,
                n.calculated_weight_grams, n.energy_kcal_total,
                n.protein_g_total, n.fat_g_total,
                n.carbohydrate_g_total
            FROM scans AS s
            LEFT JOIN scan_timings AS t ON t.scan_id = s.scan_id
            LEFT JOIN nutrition_snapshots AS n ON n.scan_id = s.scan_id
        """

    @staticmethod
    def _since_clause(since: datetime | str | None) -> tuple[str, list[str]]:
        """captured_at disimpan sebagai ISO UTC sehingga aman dibandingkan
        sebagai string selama format ``since`` sama."""
        if since is None:
            return "", []
        if isinstance(since, datetime):
            since = since.astimezone(timezone.utc).isoformat(timespec="seconds")
        return " AND s.captured_at >= ?", [str(since)]

    @staticmethod
    def _scan_clause(
        scan_id: str | None = None,
        scan_ids: list[str] | tuple[str, ...] | None = None,
    ) -> tuple[str, list[str]]:
        if scan_ids is not None:
            ids = [str(item).strip() for item in scan_ids if str(item).strip()]
            if not ids:
                return " AND 1 = 0", []
            # Hapus duplikat sambil menjaga urutan.
            seen: set[str] = set()
            unique: list[str] = []
            for value in ids:
                if value not in seen:
                    seen.add(value)
                    unique.append(value)
            placeholders = ", ".join("?" for _ in unique)
            return f" AND s.scan_id IN ({placeholders})", unique
        if scan_id is None:
            return "", []
        return " AND s.scan_id = ?", [str(scan_id)]

    def list_scans(
        self,
        limit: int | None = 100,
        offset: int = 0,
        since: datetime | str | None = None,
        scan_id: str | None = None,
        scan_ids: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        since_sql, parameters = self._since_clause(since)
        scan_sql, scan_parameters = self._scan_clause(scan_id, scan_ids)
        parameters.extend(scan_parameters)
        sql = self._select_sql() + f"""
            WHERE s.deleted_at IS NULL{since_sql}{scan_sql}
            ORDER BY s.captured_at DESC, s.created_at DESC
        """
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            parameters.extend((max(1, int(limit)), max(0, int(offset))))

        with self._reader() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        return [self._record_from_row(row) for row in rows]

    def get_scan(self, scan_id: str) -> dict | None:
        sql = self._select_sql() + """
            WHERE s.scan_id = ? AND s.deleted_at IS NULL
        """
        with self._reader() as connection:
            row = connection.execute(sql, (scan_id,)).fetchone()
            if row is None:
                return None
            detections = connection.execute(
                """
                SELECT class_id, class_name, confidence_percent,
                       x1, y1, x2, y2
                FROM detections WHERE scan_id = ? ORDER BY id
                """,
                (scan_id,),
            ).fetchall()

        record = self._record_from_row(row)
        record["detections"] = [dict(item) for item in detections]
        return record

    def count_scans(self, since: datetime | str | None = None) -> int:
        since_sql, parameters = self._since_clause(since)
        with self._reader() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM scans AS s "
                f"WHERE s.deleted_at IS NULL{since_sql}",
                parameters,
            ).fetchone()
        return int(row["count"])

    def soft_delete_all(self) -> int:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE scans SET deleted_at = ?
                WHERE deleted_at IS NULL
                """,
                (self._now(),),
            )
            return int(cursor.rowcount)

    def export_dataset(
        self,
        since: datetime | str | None = None,
        scan_id: str | None = None,
        scan_ids: list[str] | tuple[str, ...] | None = None,
    ) -> dict[str, list[dict]]:
        scans = self.list_scans(
            limit=None, since=since, scan_id=scan_id, scan_ids=scan_ids
        )
        since_sql, parameters = self._since_clause(since)
        scan_sql, scan_parameters = self._scan_clause(scan_id, scan_ids)
        parameters.extend(scan_parameters)
        with self._reader() as connection:
            detections = [
                dict(row)
                for row in connection.execute(
                    f"""
                    SELECT d.scan_id, d.class_id, d.class_name,
                           d.confidence_percent, d.x1, d.y1, d.x2, d.y2
                    FROM detections AS d
                    JOIN scans AS s ON s.scan_id = d.scan_id
                    WHERE s.deleted_at IS NULL{since_sql}{scan_sql}
                    ORDER BY s.captured_at DESC, d.id
                    """,
                    parameters,
                ).fetchall()
            ]
        return {"scans": scans, "detections": detections}

    def import_legacy_images(self, result_dir: str | Path) -> int:
        result_path = Path(result_dir).resolve()
        if not result_path.exists():
            return 0

        imported = 0
        for image_path in sorted(result_path.glob("scan_*.jpg")):
            scan_id = image_path.stem.removeprefix("scan_")
            relative_path = image_path.relative_to(self.base_dir).as_posix()
            captured_at = datetime.fromtimestamp(
                image_path.stat().st_mtime,
                timezone.utc,
            ).isoformat(timespec="seconds")
            with self._transaction() as connection:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO scans (
                        scan_id, captured_at, source, detected, food_name,
                        annotated_image_path, created_at
                    ) VALUES (?, ?, 'legacy', 0, ?, ?, ?)
                    """,
                    (
                        scan_id,
                        captured_at,
                        "Metadata lama tidak tersedia",
                        relative_path,
                        self._now(),
                    ),
                )
                imported += max(0, int(cursor.rowcount))
        return imported

    def status(self) -> dict:
        return {
            "connected": True,
            "engine": "SQLite",
            "path": str(self.database_path),
            "schema_version": self.SCHEMA_VERSION,
            "record_count": self.count_scans(),
            "size_bytes": (
                self.database_path.stat().st_size
                if self.database_path.exists()
                else 0
            ),
        }
