from __future__ import annotations

import csv
import hashlib
from pathlib import Path


class NutritionService:
    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path).resolve()
        self._records: dict[str, dict] = {}
        self.data_version = "missing"
        self.reload()

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(str(value or "").strip().lower().split())

    def reload(self) -> None:
        if not self.csv_path.exists():
            self._records = {}
            self.data_version = "missing"
            return

        raw = self.csv_path.read_bytes()
        self.data_version = hashlib.sha256(raw).hexdigest()[:12]
        records: dict[str, dict] = {}
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                name = str(row.get("Kelas_Makanan", "")).strip()
                if not name:
                    continue
                records[self._normalize(name)] = {
                    "energy_kcal": float(row["Energi_kkal_100g"]),
                    "protein_g": float(row["Protein_g_100g"]),
                    "fat_g": float(row["Lemak_g_100g"]),
                    "carbohydrate_g": float(row["Karbohidrat_g_100g"]),
                }
        self._records = records

    def _lookup(self, food_name: str) -> dict | None:
        key = self._normalize(food_name)
        if key in self._records:
            return self._records[key]
        for candidate, values in self._records.items():
            if candidate in key or key in candidate:
                return values
        return None

    def calculate(self, food_name: str, weight_grams: float) -> dict:
        per_100g = self._lookup(food_name)
        weight = max(0.0, float(weight_grams))
        if per_100g is None:
            return {
                "available": False,
                "data_version": self.data_version,
                "calculated_weight_grams": round(weight, 2),
                "per_100g": {},
                "totals": {},
            }

        multiplier = weight / 100.0
        return {
            "available": True,
            "data_version": self.data_version,
            "calculated_weight_grams": round(weight, 2),
            "per_100g": dict(per_100g),
            "totals": {
                key: round(value * multiplier, 3)
                for key, value in per_100g.items()
            },
        }
