"""Scan manual lewat terminal: pilih folder + gambar dari ``gambar/`` + input berat.

Menggunakan service yang sama dengan web (``ScanService``), jadi hasil dan
database-nya identik dengan scan dari dashboard.

Setiap scan sukses TIDAK otomatis masuk CSV. File tetap
``data/exports/hasil.csv`` hanya diisi dari scan yang di-export ke PDF
(ditulis ulang utuh dan urut per class setiap ada update).

Contoh:
    python manual_scan.py                     # pilih folder (1 / 1,3 / all), lalu pilih gambar + mode berat
    python manual_scan.py --weight-from-sensor  # berat diambil dari HX711
    python manual_scan.py --all --weight 150  # scan semua gambar (semua subfolder), berat 150 g
    python manual_scan.py --folder BASO       # scan seluruh folder BASO
    python manual_scan.py --folder BASO "Mie Ayam" --weight 150
    python manual_scan.py --images BASO/images.jfif  # satu file via path folder
    python manual_scan.py --all --weight-range 100-250  # berat acak unik per gambar
    python manual_scan.py --images nasi.jpg telur.png --weight-range 100 250
    python manual_scan.py --pdf               # pilih nomor scan bernomor (1 / 1,3 / 1-3 / all)
    python manual_scan.py --pdf all           # export PDF semua scan jadi 1 file
    python manual_scan.py --pdf 1,3           # export PDF nomor 1 dan 3 jadi 1 file
    python manual_scan.py --pdf 1-3           # export PDF nomor 1 sampai 3 jadi 1 file
    python manual_scan.py --pdf <scan_id>     # export PDF satu scan + update hasil.csv
    python manual_scan.py --pdf id1,id2       # export PDF beberapa scan_id + update hasil.csv

Mode berat untuk >1 gambar (interaktif):
    1 = sama untuk semua, 2 = acak dari range (unik, program yang menentukan),
    3 = input manual per gambar.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import time

import cv2

import config
from services.database_service import DatabaseService
from services.export_service import ExportService
from services.hx711_service import HX711Service
from services.nutrition_service import NutritionService
from services.scan_service import ScanService
from services.yolo_service import YOLOService


SOURCE = "manual_terminal"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".jfif"}
SENSOR_TIMEOUT_SECONDS = 20.0


def list_images(directory: Path, recursive: bool = False) -> list[Path]:
    if not directory.is_dir():
        return []
    if recursive:
        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def list_subfolders(directory: Path) -> list[Path]:
    """Kembalikan daftar subfolder langsung di dalam `directory`, terurut nama."""
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir() if path.is_dir()),
        key=lambda p: p.name.lower(),
    )


def _display_name(path: Path) -> str:
    """Nama tampil: 'folder/file.jpg' bila di subfolder, nama file bila di root."""
    try:
        return str(path.relative_to(config.MANUAL_IMAGE_DIR))
    except ValueError:
        return path.name


def choose_folder(base_dir: Path) -> list[Path] | None:
    """Pilih satu/beberapa folder di dalam `base_dir`.

    Contoh input: '1' / '1,3' / '1-3' / 'all'.
    Return daftar folder terpilih, atau None bila batal.
    """
    folders = list_subfolders(base_dir)
    root_images = list_images(base_dir, recursive=False)
    if not folders:
        return [base_dir] if root_images else None

    print(f"\nFolder tersedia di {base_dir}:")
    for index, folder in enumerate(folders, start=1):
        count = len(list_images(folder))
        print(f"  {index:>2}. {folder.name} ({count} gambar)")
    if root_images:
        print(f"   0. (file langsung di {base_dir.name}/) ({len(root_images)} gambar)")
    print("  Ketik nomor folder (contoh: 1 / 1,3 / 1-3 / all, q untuk batal).")

    while True:
        answer = ask("Pilih folder: ")
        cleaned = answer.strip().lower()
        if cleaned in {"q", "quit", "batal", ""} and cleaned != "":
            return None
        if not answer.strip():
            continue
        # '0' = file-file langsung di folder gambar/.
        if cleaned in {"0", ".", "root"}:
            if root_images:
                return [base_dir]
            print("Tidak ada gambar langsung di folder gambar/.")
            continue
        try:
            selected = parse_image_selection(answer, len(folders))
        except ValueError as exc:
            print(str(exc))
            continue
        if selected is None:
            return None
        return [folders[i] for i in selected]


def choose_image_source(base_dir: Path) -> list[Path]:
    """Alur 2 langkah: pilih folder dulu, lalu pilih gambar di dalamnya."""
    folders = list_subfolders(base_dir)
    # Tetap dukung layout lama (file langsung di gambar/ tanpa subfolder).
    if not folders:
        return choose_images(list_images(base_dir))
    chosen = choose_folder(base_dir)
    if not chosen:
        return []
    # Kumpulkan gambar dari folder terpilih (digabung bila pilih beberapa).
    images: list[Path] = []
    for folder in chosen:
        images.extend(list_images(folder))
    images = sorted(set(images))
    if not images:
        names = ", ".join(f.name for f in chosen)
        print(f"\nTidak ada gambar di folder: {names}.")
        return []
    if len(chosen) == 1:
        print(f"\nFolder terpilih: {chosen[0].name} ({len(images)} gambar)")
    else:
        print(
            f"\nFolder terpilih: {', '.join(f.name for f in chosen)} "
            f"({len(images)} gambar gabungan)"
        )
    return choose_images(images)


def ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def choose_image(images: list[Path]) -> Path | None:
    """Pilih satu gambar (dipertahankan untuk kompatibilitas)."""
    selected = choose_images(images)
    return selected[0] if selected else None


def parse_image_selection(answer: str, total: int) -> list[int] | None:
    """Parse input seperti '1', '1,3', '1-3', 'all'. Return index 0-based.

    Return None bila batal (q / kosong).
    Raise ValueError bila format tidak valid.
    """
    cleaned = answer.strip().lower()
    if cleaned in {"q", "quit", "batal", ""}:
        return None
    if cleaned in {"all", "*", "semua"}:
        return list(range(total))

    indices: list[int] = []
    parts = [p.strip() for p in cleaned.replace(";", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("Pilihan kosong.")
    for part in parts:
        if "-" in part:
            bounds = part.split("-", 1)
            if len(bounds) != 2 or not bounds[0].strip().isdigit() or not bounds[1].strip().isdigit():
                raise ValueError(f"Rentang tidak valid: '{part}'. Contoh: 1-3")
            start, end = int(bounds[0].strip()), int(bounds[1].strip())
            if not 1 <= start <= total and not 1 <= end <= total:
                raise ValueError(f"Rentang {start}-{end} di luar 1-{total}.")
            if start > end:
                start, end = end, start
            indices.extend(range(start - 1, end))
        else:
            if not part.isdigit() or not 1 <= int(part) <= total:
                raise ValueError(f"Nomor tidak valid: '{part}'. Pilih 1-{total}.")
            indices.append(int(part) - 1)

    # Hapus duplikat sambil menjaga urutan.
    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def choose_images(images: list[Path]) -> list[Path]:
    """Pilih satu atau beberapa gambar sekaligus.

    Contoh input: '1' / '1,3' / '1-3' / 'all'.
    """
    if not images:
        print(f"\nTidak ada gambar di {config.MANUAL_IMAGE_DIR}.")
        print("Taruh file .jpg / .jpeg / .png / .jfif di folder tersebut, lalu ulangi.")
        return []

    print(f"\nGambar tersedia ({len(images)} file):")
    for index, path in enumerate(images, start=1):
        print(f"  {index:>2}. {_display_name(path)}")

    while True:
        answer = ask("Pilih nomor gambar (contoh: 1 / 1,3 / 1-3 / all, q untuk batal): ")
        try:
            selected = parse_image_selection(answer, len(images))
        except ValueError as exc:
            print(str(exc))
            continue
        if selected is None:
            return []
        return [images[i] for i in selected]


def ask_weight() -> float | None:
    while True:
        answer = ask("Berat makanan (gram) [0]: ")
        if answer.lower() in {"q", "quit", "batal"}:
            return None
        if not answer:
            return 0.0
        try:
            weight = float(answer.replace(",", "."))
        except ValueError:
            print("Berat harus berupa angka.")
            continue
        if weight < 0:
            print("Berat tidak boleh negatif.")
            continue
        return weight


def parse_gram_range(answer: str) -> tuple[int, int]:
    """Parse range gram seperti '100-250' / '100 250' / '100,250'.

    Return (min_gram, max_gram) sebagai bilangan bulat.
    Raise ValueError bila format tidak valid.
    """
    import re

    text = answer.strip().replace(",", " ").replace(";", " ")
    # Tolak angka negatif eksplisit (diawali string / spasi / koma / titik-koma).
    # Pola digit-dash-digit seperti '100-250' dianggap pemisah range, bukan negatif.
    if re.search(r"(^|[\s,;])-\s*\d", answer.strip()):
        raise ValueError("Range gram tidak boleh negatif.")
    # Koma dianggap pemisah range (gram bulat), desimal pakai titik: 100.5-200.5
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) != 2:
        raise ValueError(
            "Format range tidak valid. Contoh yang benar: 100-250 atau 100 250"
        )
    try:
        low = int(float(numbers[0].replace(",", ".")))
        high = int(float(numbers[1].replace(",", ".")))
    except ValueError:
        raise ValueError("Range harus berupa angka. Contoh: 100-250")
    if low < 0 or high < 0:
        raise ValueError("Range gram tidak boleh negatif.")
    if low > high:
        low, high = high, low
    if low == high:
        raise ValueError(
            "Range tidak valid: batas bawah dan atas sama, "
            "berat acak tidak bisa unik."
        )
    return low, high


def ask_gram_range(count: int) -> tuple[int, int] | None:
    """Minta range gram ke user. Return None bila batal."""
    while True:
        answer = ask(
            f"Masukkan range gram untuk {count} gambar "
            "(contoh: 100-250, q untuk batal): "
        )
        if answer.lower() in {"q", "quit", "batal", ""}:
            # String kosong dianggap batal agar tidak jalan dengan range tak jelas.
            if answer.strip() == "":
                print("Range kosong, dibatalkan.")
            return None
        try:
            low, high = parse_gram_range(answer)
        except ValueError as exc:
            print(str(exc))
            continue
        span = high - low + 1
        if span < count:
            print(
                f"Range {low}-{high} hanya menyediakan {span} nilai unik, "
                f"butuh {count} nilai berbeda. Perlebar range-nya."
            )
            continue
        return low, high


def assign_random_weights(count: int, min_gram: int, max_gram: int) -> list[float]:
    """Bagi berat acak yang semuanya berbeda untuk `count` gambar.

    Menggunakan gram bulat agar jaminan unik mudah dipenuhi
    via random.sample. Return list float agar kompatibel dengan
    ScanService.record(weight_grams=...).
    """
    low, high = (min_gram, max_gram) if min_gram <= max_gram else (max_gram, min_gram)
    span = high - low + 1
    if span < count:
        raise ValueError(
            f"Range {low}-{high} hanya menyediakan {span} nilai unik, "
            f"butuh {count} nilai berbeda."
        )
    picked = random.sample(range(low, high + 1), count)
    return [float(value) for value in picked]


def sensor_weight() -> float:
    """Baca berat dari HX711; gagal dengan pesan jelas bila hardware tidak ada."""
    service = HX711Service(
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
    service.start()
    deadline = time.monotonic() + SENSOR_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            status = service.get_status()
            if status["connected"]:
                return service.get_weight()
            time.sleep(0.5)
    finally:
        service.stop()

    raise RuntimeError(
        "Sensor HX711 tidak terhubung. "
        "Jalankan tanpa --weight-from-sensor untuk input berat manual."
    )


def build_scan_service() -> ScanService:
    database_service = DatabaseService(config.DATABASE_PATH, config.BASE_DIR)
    # Bakso, capcay kuah, dan nasi goreng memakai threshold lebih rendah
    # dari global agar lebih mudah terdeteksi; kelas lain tetap memakai
    # config.YOLO_CONFIDENCE.
    class_confidences = dict(
        getattr(config, "YOLO_CLASS_CONFIDENCE", None)
        or {"bakso": 0.20, "capcay kuah": 0.15, "nasi goreng": 0.15}
    )
    return ScanService(
        YOLOService(
            model_path=config.MODEL_PATH,
            confidence=config.YOLO_CONFIDENCE,
            iou_threshold=config.YOLO_IOU_THRESHOLD,
            image_size=config.YOLO_IMAGE_SIZE,
            max_detections=config.YOLO_MAX_DETECTIONS,
            torch_threads=config.TORCH_THREADS,
            class_confidences=class_confidences,
        ),
        NutritionService(config.NUTRITION_CSV_PATH),
        database_service,
        base_dir=config.BASE_DIR,
        photo_dir=config.PHOTO_DIR,
    )


def print_result(record: dict) -> None:
    nutrition = record.get("nutrition") or {}
    totals = nutrition.get("totals") or {}
    print("\n=== Hasil scan ===")
    print(f"  Scan ID       : {record['scan_id']}")
    print(f"  Sumber        : {record['source']}")
    print(f"  Deteksi       : {record['detected']}")
    print(f"  Makanan       : {record['food_name']}")
    print(f"  Confidence    : {record['confidence']:.2f}%")
    print(f"  Jumlah objek  : {record['object_count']}")
    print(f"  Berat         : {record['weight_grams']:.2f} g")
    print(f"  Energi        : {float(totals.get('energy_kcal') or 0):.2f} kkal")
    print(f"  Protein       : {float(totals.get('protein_g') or 0):.2f} g")
    print(f"  Lemak         : {float(totals.get('fat_g') or 0):.2f} g")
    print(f"  Karbohidrat    : {float(totals.get('carbohydrate_g') or 0):.2f} g")
    if not nutrition.get("available"):
        print("  Catatan       : data nutrisi untuk makanan ini tidak ada di csv.csv")
    print(f"  Foto asli     : {record['original_image_path']}")
    print(f"  Foto hasil    : {record['annotated_image_path']}")


def print_scan_choices(scans: list[dict], limit: int = 50) -> None:
    """Tampilkan daftar scan bernomor untuk dipilih (terbaru dulu)."""
    total = len(scans)
    shown = scans[:limit] if total > limit else scans
    print(f"\nRiwayat scan ({total} data, terbaru dulu):")
    for index, scan in enumerate(shown, start=1):
        print(
            f"  {index:>3}. {scan.get('scan_id')} | "
            f"{scan.get('timestamp')} | "
            f"{scan.get('food_name') or '-'} | "
            f"{float(scan.get('weight_grams') or 0):.0f} g"
        )
    if total > limit:
        print(
            f"  ... hanya {limit} terbaru ditampilkan. "
            f"Nomor 1-{total} tetap bisa dipakai "
            f"(contoh: --pdf \"{total}\" untuk data tertua)."
        )


def unique_foods(scans: list[dict]) -> list[str]:
    """Kembalikan daftar food_name unik sesuai urutan kemunculan.

    Perbandingan case-insensitive, tampilan memakai ejaan pertama ditemukan.
    """
    seen: dict[str, str] = {}
    for scan in scans:
        raw = str(scan.get("food_name") or "-").strip() or "-"
        key = raw.lower()
        if key not in seen:
            seen[key] = raw
    return list(seen.values())


def print_food_choices(scans: list[dict]) -> list[str]:
    """Tampilkan daftar class makanan + jumlah scan. Return daftar class."""
    foods = unique_foods(scans)
    print(f"\nClass terdeteksi ({len(foods)} class, {len(scans)} scan):")
    for number, food in enumerate(foods, start=1):
        positions = [
            str(i + 1) for i, s in enumerate(scans)
            if str(s.get("food_name") or "-").strip().lower() == food.lower()
        ]
        print(f"  {number:>2}. {food} ({len(positions)} scan: no {', '.join(positions)})")
    return foods


def parse_food_selection(answer: str, foods: list[str]) -> list[int] | None:
    """Parse pilihan class: '1', '1,2', '1-2', 'all', atau nama class.

    Contoh nama: 'mie kocok' / 'mie kocok,bakso'.
    Return index 0-based ke `foods`. None bila batal.
    Raise ValueError bila tidak valid.
    """
    cleaned = answer.strip().lower()
    if cleaned in {"q", "quit", "batal", ""}:
        return None
    if cleaned in {"all", "*", "semua"}:
        return list(range(len(foods)))
    by_name = {food.lower(): idx for idx, food in enumerate(foods)}
    tokens = [t.strip() for t in cleaned.replace(";", ",").split(",") if t.strip()]
    if not tokens:
        raise ValueError("Pilihan kosong.")
    indices: list[int] = []
    for token in tokens:
        # Rentang nomor class: '1-3'.
        if "-" in token and all(p.strip().isdigit() for p in token.split("-", 1) if p.strip()):
            bounds = token.split("-", 1)
            start, end = int(bounds[0].strip()), int(bounds[1].strip())
            if not 1 <= start <= len(foods) or not 1 <= end <= len(foods):
                raise ValueError(f"Rentang {start}-{end} di luar 1-{len(foods)}.")
            if start > end:
                start, end = end, start
            indices.extend(range(start - 1, end))
            continue
        if token.isdigit():
            if not 1 <= int(token) <= len(foods):
                raise ValueError(f"Nomor class tidak valid: '{token}'. Pilih 1-{len(foods)}.")
            indices.append(int(token) - 1)
            continue
        # Nama class: cocok persis dulu (case-insensitive), lalu substring unik.
        if token in by_name:
            indices.append(by_name[token])
            continue
        matches = [idx for idx, food in enumerate(foods) if token in food.lower()]
        if len(matches) == 1:
            indices.append(matches[0])
        elif len(matches) > 1:
            names = ", ".join(foods[i] for i in matches)
            raise ValueError(f"'{token}' cocok ke beberapa class: {names}. Ketik nama lengkap.")
        else:
            names = ", ".join(foods)
            raise ValueError(f"Class tidak dikenal: '{token}'. Pilihan: {names}.")
    seen: set[int] = set()
    unique: list[int] = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            unique.append(i)
    return unique


def filter_scans_by_foods(scans: list[dict], foods: list[str], indices: list[int]) -> list[dict]:
    """Kembalikan semua scan yang food_name-nya termasuk class terpilih."""
    wanted = {foods[i].lower() for i in indices}
    return [
        scan for scan in scans
        if str(scan.get("food_name") or "-").strip().lower() in wanted
    ]


def choose_scans_by_food(scans: list[dict]) -> list[dict] | None:
    """Minta user memilih class. Return None bila batal."""
    foods = print_food_choices(scans)
    while True:
        answer = ask(
            "Pilih class yang mau di-export "
            "(contoh: 1 / 1,2 / mie kocok / all, q untuk batal): "
        )
        try:
            indices = parse_food_selection(answer if answer.strip() else "all", foods)
        except ValueError as exc:
            print(str(exc))
            continue
        if indices is None:
            return None
        return filter_scans_by_foods(scans, foods, indices)


def choose_scans_interactive(scans: list[dict]) -> list[dict]:
    """Minta user memilih scan. Default: pilih per class, bukan per nomor."""
    if not scans:
        print("\nBelum ada data scan di database.")
        return []
    print_scan_choices(scans)
    print_food_choices(scans)
    while True:
        mode = ask(
            "Export berdasarkan [1=nomor scan / 2=nama class] (default 2, q untuk batal): "
        ).strip().lower()
        if mode in {"q", "quit", "batal"}:
            return []
        if not mode:
            mode = "2"
        if mode not in {"1", "2"}:
            print("Pilih 1 atau 2.")
            continue
        break
    if mode == "2":
        chosen = choose_scans_by_food(scans)
        return chosen if chosen is not None else []
    while True:
        answer = ask(
            "Pilih nomor yang mau di-export "
            "(contoh: 1 / 1,3 / 1-3 / all, q untuk batal): "
        )
        try:
            indices = parse_image_selection(answer, len(scans))
        except ValueError as exc:
            print(str(exc))
            continue
        if indices is None:
            return []
        return [scans[i] for i in indices]


def resolve_scan_selection(target: str, scans: list[dict]) -> list[dict]:
    """Ubah argumen --pdf menjadi daftar scan terpilih.

    Mendukung: 'all', 'pilih' (interaktif), nomor ('1,3' / '1-3'),
    nama class ('mie kocok' / 'mie kocok,bakso'),
    satu scan_id, atau gabungan ('id1,id2' / '1,id2').
    Raise SystemExit bila tidak valid.
    """
    import re

    cleaned = (target or "").strip()
    lowered = cleaned.lower()
    if lowered in {"all", "*", "semua"}:
        return list(scans)
    if lowered in {"pilih", "select", "pilihan", "choice"}:
        return choose_scans_interactive(scans)
    if not scans:
        raise SystemExit("Belum ada data scan di database.")
    # Nama class langsung: '--pdf "mie kocok"' -> semua scan mie kocok.
    # '--pdf "mie kocok,bakso"' -> semua scan kedua class itu.
    foods = unique_foods(scans)
    try:
        # Hanya coba bila token mengandung huruf (bukan murni nomor/scan_id).
        if re.search(r"[a-zA-Z]", cleaned):
            food_indices = parse_food_selection(cleaned, foods)
            if food_indices:
                matched = filter_scans_by_foods(scans, foods, food_indices)
                if matched:
                    return matched
    except ValueError:
        pass  # Bukan nama class valid -> lanjut ke parsing nomor/scan_id.
    # Normalisasi pemisah: spasi/titik-koma dianggap koma, '1 - 3' -> '1-3'.
    # Ini juga menangani PowerShell yang memecah 1,3 menjadi ['1', '3'].
    normalized = re.sub(r"\s*-\s*", "-", cleaned)
    normalized = re.sub(r"[\s;]+", ",", normalized)
    normalized = re.sub(r",+", ",", normalized).strip(",")
    # Murni nomor: '1', '1,3', '1-3', '1 3'.
    if re.fullmatch(r"[\d\s,;\-]+", cleaned) and re.search(r"\d", cleaned):
        try:
            indices = parse_image_selection(normalized, len(scans))
        except ValueError as exc:
            raise SystemExit(str(exc))
        if indices is None:
            return []
        return [scans[i] for i in indices]
    # Gabungan dipisah koma: tiap token boleh nomor atau scan_id.
    if "," in cleaned or ";" in cleaned:
        tokens = [
            part.strip()
            for part in cleaned.replace(";", ",").split(",")
            if part.strip()
        ]
        selected: list[dict] = []
        by_id = {str(scan.get("scan_id")): scan for scan in scans}
        for token in tokens:
            if token.isdigit() and 1 <= int(token) <= len(scans):
                selected.append(scans[int(token) - 1])
            elif token in by_id:
                selected.append(by_id[token])
            else:
                raise SystemExit(
                    f"Pilihan tidak dikenal: '{token}'. "
                    f"Gunakan nomor 1-{len(scans)}, scan_id, atau 'all'."
                )
        # Hapus duplikat sambil menjaga urutan.
        seen: set[str] = set()
        unique: list[dict] = []
        for scan in selected:
            key = str(scan.get("scan_id"))
            if key not in seen:
                seen.add(key)
                unique.append(scan)
        return unique
    # Satu scan_id.
    for scan in scans:
        if str(scan.get("scan_id")) == cleaned:
            return [scan]
    raise SystemExit(
        f"Scan ID / nomor tidak ditemukan: {cleaned}. "
        "Gunakan --pdf tanpa nilai untuk melihat daftar bernomor."
    )


def _join_export_target(value: str | list[str] | None) -> str:
    """Satukan nilai argparse --pdf (nargs='*') jadi satu string.

    None (flag tidak dipakai) tidak terjadi di sini. [] (flag tanpa nilai)
    berarti mode interaktif 'pilih'. List ['1', '3'] (PowerShell tanpa kutip)
    digabung menjadi '1,3'.
    """
    if value is None:
        return "pilih"
    if isinstance(value, str):
        return value if value.strip() else "pilih"
    items = [str(item).strip() for item in value if str(item).strip()]
    if not items:
        return "pilih"
    # Gabung dengan koma agar '1 3' dan '1,3' diperlakukan sama.
    return ",".join(items)


def export_pdf(target: str) -> list[Path]:
    """Export PDF sekaligus update hasil.csv hanya dari scan terpilih.

    hasil.csv digabung (append) lalu ditulis ulang urut per class,
    jadi isinya hanya scan yang pernah di-export ke PDF.
    """
    database_service = DatabaseService(config.DATABASE_PATH, config.BASE_DIR)
    export_service = ExportService(database_service)
    scans = database_service.list_scans(limit=None)
    selected = resolve_scan_selection(target, scans)
    if not selected:
        print("Dibatalkan.")
        raise SystemExit(1)

    config.EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    # 'all' eksplisit tetap 1 PDF gabungan (perilaku lama).
    # Pilihan nomor/scan_id (satu atau banyak) -> 1 PDF terpisah per scan.
    if (target or "").strip().lower() in {"all", "*", "semua"}:
        artifact = export_service.create("pdf")
        destination = config.EXPORT_DIR / artifact.filename
        destination.write_bytes(artifact.content.getvalue())
        print(f"\nPDF tersimpan: {destination} ({len(selected)} scan)")
    else:
        destinations: list[Path] = []
        for scan in selected:
            scan_id = str(scan.get("scan_id"))
            artifact = export_service.create("pdf", scan_id=scan_id)
            destination = config.EXPORT_DIR / f"scan-{scan_id}.pdf"
            destination.write_bytes(artifact.content.getvalue())
            destinations.append(destination)

        print(f"\n{len(destinations)} PDF terpisah tersimpan di {config.EXPORT_DIR}:")
        for scan, destination in zip(selected, destinations):
            print(f"  - {scan.get('scan_id')}: {scan.get('food_name')} -> {destination.name}")
        destination = destinations[0] if len(destinations) == 1 else config.EXPORT_DIR

    try:
        hasil_path = export_service.update_hasil_csv(
            selected,
            config.HASIL_CSV_PATH,
            legacy_path=config.LEGACY_CSV_PATH,
        )
        print(f"  hasil.csv diperbarui dari {len(selected)} scan PDF: {hasil_path}")
    except Exception as exc:
        print(f"  Gagal menulis hasil.csv: {exc}")
    if (target or "").strip().lower() in {"all", "*", "semua"}:
        return [destination]
    return destinations


def run_interactive(
    scan_service: ScanService,
    use_sensor: bool,
    preselected: list[Path] | None = None,
    preset_weight: float | None = None,
    preset_range: tuple[int, int] | None = None,
) -> list[dict] | None:
    if preselected is not None:
        image_paths = preselected
    else:
        image_paths = choose_image_source(config.MANUAL_IMAGE_DIR)
    if not image_paths:
        print("Dibatalkan.")
        return None

    weights: list[float] = []
    if preset_weight is not None:
        weights = [preset_weight] * len(image_paths)
        print(f"\nBerat untuk semua gambar: {preset_weight:.2f} g")
    elif use_sensor:
        print("Berat diambil dari HX711 per gambar.")
        for image_path in image_paths:
            input(f"Taruh '{_display_name(image_path)}' di timbangan lalu tekan Enter... ")
            print("Membaca berat dari HX711...")
            weight = sensor_weight()
            print(f"Berat terbaca untuk {_display_name(image_path)}: {weight:.2f} g")
            weights.append(weight)
    elif preset_range is not None:
        # Mode non-interaktif: range dari CLI, acaknya oleh program (unik).
        low, high = preset_range
        try:
            weights = assign_random_weights(len(image_paths), low, high)
        except ValueError as exc:
            print(str(exc))
            return None
        print(f"\nBerat acak (unik) dari range {low}-{high} g:")
        for path, w in zip(image_paths, weights):
            print(f"  - {_display_name(path)}: {w:.0f} g")
    elif len(image_paths) == 1:
        weight = ask_weight()
        if weight is None:
            print("Dibatalkan.")
            return None
        weights = [weight]
    else:
        print(f"\n{len(image_paths)} gambar dipilih:")
        for index, path in enumerate(image_paths, start=1):
            print(f"  {index:>2}. {_display_name(path)}")
        print("\nPilih mode berat:")
        print("  1. Sama untuk semua gambar")
        print("  2. Acak dari range (unik per gambar, program yang menentukan)")
        print("  3. Input manual per gambar")
        while True:
            mode = ask("Pilih [1/2/3, q untuk batal] (default 2): ")
            if mode.lower() in {"q", "quit", "batal"}:
                print("Dibatalkan.")
                return None
            if not mode:
                mode = "2"
            if mode in {"1", "2", "3"}:
                break
            print("Pilih 1, 2, atau 3.")
        if mode == "1":
            weight = ask_weight()
            if weight is None:
                print("Dibatalkan.")
                return None
            weights = [weight] * len(image_paths)
        elif mode == "2":
            gram_range = ask_gram_range(len(image_paths))
            if gram_range is None:
                print("Dibatalkan.")
                return None
            low, high = gram_range
            weights = assign_random_weights(len(image_paths), low, high)
            print(f"\nBerat acak (unik) dari range {low}-{high} g:")
            for path, w in zip(image_paths, weights):
                print(f"  - {_display_name(path)}: {w:.0f} g")
        else:
            for image_path in image_paths:
                print(f"\nBerat untuk {_display_name(image_path)}:")
                weight = ask_weight()
                if weight is None:
                    print("Dibatalkan.")
                    return None
                weights.append(weight)

    if preselected is None and preset_weight is None and preset_range is None and not use_sensor:
        summary = ", ".join(
            f"{_display_name(path)} ({w:.0f} g)" for path, w in zip(image_paths, weights)
        )
        confirm = ask(f"\nScan {len(image_paths)} gambar ini? [{summary}] [Y/n]: ")
        if confirm.lower() in {"n", "no", "batal"}:
            print("Dibatalkan.")
            return None

    records: list[dict] = []
    for position, (image_path, weight) in enumerate(zip(image_paths, weights), start=1):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[{position}/{len(image_paths)}] Gambar gagal dibaca, dilewati: {image_path}")
            continue

        print(f"\n[{position}/{len(image_paths)}] Memproses {_display_name(image_path)} ({weight:.2f} g)...")
        try:
            record = scan_service.record(
                frame,
                source=SOURCE,
                weight_grams=weight,
            )
        except Exception as exc:
            print(f"  Gagal scan {_display_name(image_path)}: {exc}")
            continue
        print_result(record)
        records.append(record)

    if not records:
        print("\nTidak ada gambar yang berhasil di-scan.")
        return None

    print(f"\n=== Ringkasan batch: {len(records)}/{len(image_paths)} berhasil ===")
    for index, record in enumerate(records, start=1):
        print(
            f"  {index:>2}. {record['scan_id']}: "
            f"{record['food_name']} ({record['weight_grams']:.0f} g)"
        )
    print(f"\nHasil scan tersimpan di database (belum masuk hasil.csv).")
    print(f"hasil.csv hanya diisi dari scan yang di-export ke PDF: {config.HASIL_CSV_PATH}")

    pilihan = ask("\nExport PDF hasil batch ini? [pdf/Skip]: ").strip().lower()
    if pilihan in {"pdf", "p", "y", "ya", "yes"}:
        foods = print_food_choices(records)
        while True:
            mode = ask(
                "Export berdasarkan [1=nomor scan / 2=nama class] (default 2, q untuk batal): "
            ).strip().lower()
            if mode in {"q", "quit", "batal"}:
                print("Export batch dibatalkan.")
                return records
            if not mode:
                mode = "2"
            if mode in {"1", "2"}:
                break
            print("Pilih 1 atau 2.")
        chosen: list[dict] = []
        if mode == "2":
            while True:
                jawaban = ask(
                    "Class mana yang di-export? "
                    "(contoh: 1 / 1,2 / mie kocok / all, q untuk batal): "
                )
                try:
                    food_indices = parse_food_selection(
                        jawaban if jawaban.strip() else "all", foods
                    )
                except ValueError as exc:
                    print(str(exc))
                    continue
                if food_indices is None:
                    print("Export batch dibatalkan.")
                    break
                chosen = filter_scans_by_foods(records, foods, food_indices)
                break
        else:
            while True:
                nomor = ask(
                    "Nomor batch mana yang di-export? "
                    "(contoh: 1 / 1,3 / 1-3 / all, q untuk batal): "
                )
                try:
                    indices = parse_image_selection(
                        nomor if nomor.strip() else "all", len(records)
                    )
                except ValueError as exc:
                    print(str(exc))
                    continue
                if indices is None:
                    print("Export batch dibatalkan.")
                    break
                chosen = [records[i] for i in indices]
                break
        if chosen:
            export_pdf(",".join(r["scan_id"] for r in chosen))
            print(f"{len(chosen)} PDF terpisah tersimpan di {config.EXPORT_DIR} + hasil.csv diperbarui.")

    return records


def resolve_cli_images(
    names: list[str] | None,
    scan_all: bool,
    folders: list[str] | None = None,
) -> list[Path] | None:
    """Kembalikan daftar gambar dari argumen CLI, atau None bila mode interaktif.

    Mendukung path subfolder, mis. 'BASO/images.jfif' atau 'BASO',
    dan nomor urut dari daftar rekursif.
    """
    available = list_images(config.MANUAL_IMAGE_DIR, recursive=True)
    if folders:
        picked: list[Path] = []
        for folder_name in folders:
            folder_path = (config.MANUAL_IMAGE_DIR / folder_name.strip()).resolve()
            try:
                folder_path.relative_to(config.MANUAL_IMAGE_DIR.resolve())
            except ValueError:
                raise SystemExit(f"Folder di luar gambar/: {folder_name}.")
            if not folder_path.is_dir():
                raise SystemExit(
                    f"Folder tidak ditemukan: {folder_name}. "
                    "Lihat daftar folder dengan: python manual_scan.py (tanpa argumen)."
                )
            picked.extend(list_images(folder_path))
        return sorted(set(picked))
    if scan_all:
        return available
    if not names:
        return None
    by_name = {path.name.lower(): path for path in available}
    by_relative = {
        str(path.relative_to(config.MANUAL_IMAGE_DIR)).lower().replace("\\", "/"): path
        for path in available
    }
    resolved: list[Path] = []
    for name in names:
        cleaned = name.strip().strip("\"'")
        candidate = Path(cleaned)
        # File langsung (path absolut / relatif dari cwd).
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            resolved.append(candidate)
            continue
        # Relatif ke folder gambar/: 'BASO/images.jfif' atau 'BASO'.
        rel_key = cleaned.lower().replace("\\", "/").lstrip("./")
        if rel_key in by_relative:
            resolved.append(by_relative[rel_key])
            continue
        direct = config.MANUAL_IMAGE_DIR / cleaned
        if direct.is_dir():
            resolved.extend(list_images(direct))
            continue
        if direct.is_file() and direct.suffix.lower() in IMAGE_SUFFIXES:
            resolved.append(direct)
            continue
        match = by_name.get(candidate.name.lower())
        if match is not None:
            # Bila nama file duplikat di beberapa folder, minta path folder.
            twins = [p for p in available if p.name.lower() == candidate.name.lower()]
            if len(twins) > 1:
                choices = ", ".join(
                    str(p.relative_to(config.MANUAL_IMAGE_DIR)) for p in twins
                )
                raise SystemExit(
                    f"Nama '{candidate.name}' ada di beberapa folder: {choices}. "
                    "Gunakan path folder, mis. --images BASO/images.jfif."
                )
            resolved.append(match)
            continue
        # Dukung nomor urut seperti --images 1 3
        if cleaned.isdigit() and 1 <= int(cleaned) <= len(available):
            resolved.append(available[int(cleaned) - 1])
            continue
        raise SystemExit(
            f"Gambar/folder tidak ditemukan: {name}. "
            f"Contoh: --folder BASO / --images BASO/images.jfif / --all."
        )
    # Hapus duplikat sambil menjaga urutan.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in resolved:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan manual lewat terminal ke database yang sama dengan web."
    )
    parser.add_argument(
        "--weight-from-sensor",
        action="store_true",
        help="Ambil berat dari sensor HX711, bukan input manual.",
    )
    parser.add_argument(
        "--pdf",
        nargs="*",
        default=None,
        metavar="PILIHAN",
        help=(
            "Export PDF + update hasil.csv lalu keluar. Tanpa nilai = pilih "
            "interaktif (nomor/class). "
            'Contoh: --pdf all / --pdf "1,3" / --pdf 1-3 / '
            '"--pdf mie kocok" / --pdf <scan_id> / --pdf id1,id2. '
            "Di PowerShell gunakan tanda kutip untuk koma: --pdf \"1,3\"."
        ),
    )
    parser.add_argument(
        "--images",
        nargs="+",
        metavar="NAMA_FILE",
        help=(
            "Scan langsung tanpa menu. Contoh: --images BASO/images.jfif "
            "atau --images BASO (seluruh folder) atau --images 1 3"
        ),
    )
    parser.add_argument(
        "--folder",
        nargs="+",
        metavar="NAMA_FOLDER",
        help=(
            "Scan seluruh gambar di subfolder gambar/. Contoh: "
            "--folder BASO atau --folder BASO \"Mie Ayam\"."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scan semua gambar di folder gambar/ sekaligus (termasuk semua subfolder).",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="Berat (gram) yang dipakai untuk semua gambar. Tanpa ini akan ditanya.",
    )
    parser.add_argument(
        "--weight-range",
        nargs="+",
        metavar="GRAM",
        default=None,
        help=(
            "Range gram untuk berat acak unik per gambar. "
            'Contoh: --weight-range 100-250 atau --weight-range 100 250. '
            "Program mengacak nilainya, semuanya dijamin berbeda."
        ),
    )
    args = parser.parse_args()

    try:
        if args.pdf is not None:
            export_pdf(_join_export_target(args.pdf))
            return 0
        if args.weight is not None and args.weight < 0:
            raise SystemExit("Berat (--weight) tidak boleh negatif.")
        if args.weight_from_sensor and args.weight is not None:
            raise SystemExit("Pilih salah satu: --weight-from-sensor atau --weight.")
        weight_range: tuple[int, int] | None = None
        if args.weight_range is not None:
            if args.weight is not None or args.weight_from_sensor:
                raise SystemExit(
                    "Pilih salah satu: --weight, --weight-from-sensor, atau --weight-range."
                )
            try:
                weight_range = parse_gram_range(" ".join(args.weight_range))
            except ValueError as exc:
                raise SystemExit(f"--weight-range tidak valid: {exc}")
        preselected = resolve_cli_images(args.images, args.all, folders=args.folder)
        if preselected is not None and not preselected:
            print(f"Tidak ada gambar di {config.MANUAL_IMAGE_DIR}.")
            return 1
        if weight_range is not None and preselected is not None:
            low, high = weight_range
            if (high - low + 1) < len(preselected):
                raise SystemExit(
                    f"Range {low}-{high} hanya menyediakan {high - low + 1} nilai unik, "
                    f"butuh {len(preselected)} gambar berbeda. Perlebar range-nya."
                )
        records = run_interactive(
            build_scan_service(),
            use_sensor=args.weight_from_sensor,
            preselected=preselected,
            preset_weight=args.weight,
            preset_range=weight_range,
        )
        return 0 if records else 1
    except KeyboardInterrupt:
        print("\nDibatalkan.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
