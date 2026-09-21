# 🥗 FoodVision AI Dashboard

**FoodVision AI Dashboard** adalah sistem dashboard cerdas berbasis **Raspberry Pi**, **Flask**, **YOLOv5 (ONNX/PyTorch)**, dan sensor timbangan **HX711 (Load Cell)** untuk pengenalan jenis makanan sekaligus penghitungan nilai nutrisi (kalori, protein, lemak, dan karbohidrat) secara real-time berdasarkan berat aktual porsi makanan.

Proyek ini dirancang hemat daya dan sumber daya, dengan pemrosesan inferensi AI on-demand (hanya saat tombol scan ditekan), sehingga video feed kamera tetap lancar tanpa membebani prosesor Raspberry Pi secara terus-menerus.

---

## 🌟 Fitur Utama

- **Live Camera Stream (MJPEG)**: Preview kamera real-time yang ringan tanpa lag inferensi kontinu.
- **On-Demand YOLOv5 Inference**: Deteksi objek makanan dijalankan hanya saat diminta (*single-shot inference*), menghemat beban CPU & RAM.
- **Timbangan Presisi HX711**: Pembacaan berat real-time dilengkapi fitur **Tare / Zeroing** dan filter *Moving Average* untuk meminimalisasi noise / getaran.
- **Analisis Nutrisi Otomatis**: Integrasi otomatis antara hasil deteksi makanan dan berat gram aktual dengan database gizi (`csv.csv`).
- **Penyimpanan Permanen (SQLite & Foto)**: Setiap pemindaian tercatat di database SQLite lokal (`data/food_recognition.db`) dan arsip foto (original & annotated) tersusun rapi berdasarkan tanggal.
- **Export Laporan Lengkap**: Ekspor data riwayat ke format **CSV**, **Excel (XLSX)**, dan **PDF** langsung dari browser.
- **Hardware Fallback & Mocking**: Sistem tetap dapat berjalan di lingkungan pengujian/laptop non-Raspberry Pi melalui mode simulasi hardware otomatis.

---

## 📂 Struktur Direktori Proyek

```text
food_ai_dashboard/
├── app.py                     # Entry point aplikasi Flask & routing API
├── config.py                  # Konfigurasi global (kamera, pin GPIO, path, YOLO, dll.)
├── run.sh                     # Script shell runner (otomatis aktivasi virtualenv)
├── requirements.txt           # Kebutuhan library Python
├── best.onnx / best.pt        # Model deteksi YOLOv5
├── classes.txt                # Daftar nama kelas makanan
├── csv.csv                    # Database kandungan nutrisi per 100 gram
├── images.jpg                 # Gambar sampel untuk pengujian offline
│
├── services/                  # Business logic modular
│   ├── camera_service.py      # Pengendali stream dan capture frame kamera
│   ├── database_service.py    # Handler database SQLite dan manajemen riwayat
│   ├── export_service.py      # Generator file export (CSV, XLSX, PDF)
│   ├── hx711_service.py       # Handler sensor timbangan HX711 & tare
│   ├── label_mapping.py       # Mapping label deteksi ke kelas nutrisi
│   └── yolo_service.py        # Pipeline inferensi YOLOv5 (ONNX Runtime / PyTorch)
│
├── static/                    # Asset web statis (CSS, JS, hasil scan)
│   ├── app.js                 # Logika interaktif frontend (AJAX, chart, tare, scan)
│   ├── style.css              # Tampilan dashboard modern & responsif
│   └── results/               # Direktori sementara output gambar hasil scan
│
├── templates/                 # Template HTML UI
│   └── index.html             # Tampilan dashboard utama
│
├── data/                      # Penyimpanan data lokal
│   ├── food_recognition.db    # Database SQLite riwayat scan
│   ├── scan_history.csv       # File log riwayat alternatif (CSV)
│   └── photos/                # Arsip foto scan (YYYY/MM/DD/SCAN_ID/)
│
├── deploy/                    # File konfigurasi deployment
│   └── food-recognition.service # Konfigurasi auto-start systemd di Linux/RPi
│
└── tests/                     # Unit test otomatis
    ├── test_camera_service.py
    ├── test_database_service.py
    ├── test_hx711_service.py
    └── test_yolo_service.py
```

---

## ⚙️ Kebutuhan Sistem & Hardware

### Perangkat Keras:
- **Komputer Mini**: Raspberry Pi 3 Model B+ / 4 / 5 (Raspberry Pi OS Bookworm / Bullseye 64-bit disarankan).
- **Kamera**: USB Webcam (UVC) atau Raspberry Pi Camera Module.
- **Sensor Timbangan**: Load Cell (1kg / 5kg) terhubung ke modul amplifier **HX711**.
- **Koneksi Pin Default HX711 (GPIO BCM)**:
  - `VCC` -> `5V` (Pin 2/4) atau `3.3V` (Pin 1)
  - `GND` -> `GND` (Pin 6/9/14/20/...)
  - `DT / DOUT` -> `GPIO 5` (Pin 29)
  - `SCK` -> `GPIO 6` (Pin 31)

### Perangkat Lunak:
- Python 3.9+ (disarankan Python 3.10 / 3.11)
- Virtual environment (`venv`)

---

## 🚀 Panduan Instalasi

### 1. Unduh atau Clone Proyek
Letakkan direktori proyek pada folder kerja Anda:
```bash
cd ~
# atau ekstrak zip jika diunduh manual
cd food_ai_dashboard
```

### 2. Siapkan Virtual Environment
Gunakan opsi `--system-site-packages` terutama di Raspberry Pi agar OpenCV / GPIO bawaan sistem dapat terdeteksi:
```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
```

### 3. Pasang Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Jika `onnxruntime` belum terpasang di Raspberry Pi Anda:
```bash
pip install --no-cache-dir onnxruntime
```

### 4. Persiapan Model & Kelas
1. Pastikan model terlatih Anda berada di root folder dengan nama `best.onnx` (atau `best.pt`).
2. Sesuaikan file `classes.txt` dengan daftar kelas dataset Anda (urutan harus sama persis dengan `data.yaml` saat pelatihan YOLO):
   ```text
   nasi_goreng
   mie_goreng
   sate
   ayam_goreng
   ```
3. Pastikan `csv.csv` memuat data nutrisi per 100 gram untuk tiap kelas makanan.

---

## 🔧 Konfigurasi (`config.py`)

Pengaturan parameter utama dapat disesuaikan pada file `config.py`:

```python
# Model & Kamera
MODEL_PATH = BASE_DIR / "best.pt"      # atau "best.onnx"
CAMERA_INDEX = 0                      # Indeks /dev/videoX
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# YOLO
YOLO_CONFIDENCE = 0.25                # Batas confidence score
YOLO_IOU_THRESHOLD = 0.45             # Ambang batas NMS
TORCH_THREADS = 2                     # Pembatasan thread CPU untuk Raspberry Pi

# Timbangan HX711
HX711_DT_PIN = 5                      # Pin DT
HX711_SCK_PIN = 6                     # Pin SCK
HX711_CALIBRATION_FACTOR = 718.42     # Faktor pengali kalibrasi timbangan

# Web Server
HOST = "0.0.0.0"
PORT = 5000
```

---

## 🏃 Cara Menjalankan Aplikasi

### Metode 1: Menjalankan Langsung (Development / Debugging)
```bash
source .venv/bin/activate
python app.py
```

### Metode 2: Menggunakan Script Runner
```bash
chmod +x run.sh
./run.sh
```

### Akses Dashboard
Buka browser di perangkat lokal atau laptop yang terhubung ke jaringan Wi-Fi/LAN yang sama:
- **Dari Raspberry Pi**: `http://127.0.0.1:5000`
- **Dari Perangkat Lain**: `http://<IP-RASPBERRY-PI>:5000` *(cek IP dengan perintah `hostname -I`)*

---

## 🔄 Deployment Otomatis (Systemd Service)

Agar dashboard otomatis berjalan saat Raspberry Pi dinyalakan:

1. Salin dan sesuaikan file service:
   ```bash
   sudo cp deploy/food-recognition.service /etc/systemd/system/
   ```
2. Sesuaikan path user dan direktori pada `/etc/systemd/system/food-recognition.service` jika berbeda.
3. Reload daemon dan aktifkan service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable food-recognition.service
   sudo systemctl start food-recognition.service
   ```
4. Cek status service:
   ```bash
   sudo systemctl status food-recognition.service
   ```

---

## 📖 Panduan Penggunaan Dashboard

1. **Persiapan Awal**: Pastikan timbangan dalam kondisi kosong saat dashboard baru dibuka.
2. **Tare / Zero**: Klik tombol **Tare / Zero** pada dashboard jika timbangan menunjukkan angka bukan 0 gram saat kosong.
3. **Posisikan Makanan**: Letakkan piring/makanan di atas timbangan dan pastikan objek terlihat jelas pada live preview kamera.
4. **Mulai Scan**: Klik tombol **Mulai Scan**.
   - Kamera mengambil 1 snapshot frame.
   - Sensor mencatat berat porsi saat pemindaian.
   - Model YOLO mendeteksi jenis makanan dan bounding box.
   - Sistem mengkalkulasi nutrisi sesuai porsi berat dan menampilkan hasilnya di layar.
5. **Riwayat & Laporan**: Riwayat pemindaian dapat difilter berdasarkan rentang waktu (`6h`, `24h`, `7d`, `30d`, `Semua`) dan diunduh dalam format **CSV**, **Excel**, atau **PDF**.

---

## 🔌 Daftar Endpoint REST API

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Antarmuka web dashboard utama |
| `GET` | `/video_feed` | Stream video kamera MJPEG |
| `GET` | `/api/status` | Status kesehatan sistem, kamera, model, dan timbangan |
| `POST` | `/api/scan` | Memicu proses snapshot, pembacaan timbangan, dan deteksi |
| `GET` | `/api/weight` | Pembacaan nilai berat terkini dari sensor HX711 |
| `POST` | `/api/tare` | Melakukan reset kalibrasi titik nol (*tare*) timbangan |
| `GET` | `/api/history` | Mengambil data riwayat scan (query: `range`, `limit`, `offset`) |
| `POST` | `/api/history/clear`| Menghapus tampilan riwayat scan (*soft delete*) |
| `GET` | `/api/scans/<id>` | Detail data scan spesifik beserta informasi nutrisi |
| `GET` | `/api/scans/<id>/image/<type>` | Mengambil gambar (`original` atau `annotated`) |
| `GET` | `/api/export/csv` | Mengunduh file export riwayat format CSV |
| `GET` | `/api/export/xlsx` | Mengunduh file export riwayat format Excel |
| `GET` | `/api/export/pdf` | Mengunduh file laporan cetak format PDF |

---

## 🛠️ Troubleshooting & Tips

### 1. Kamera Timeout atau Tidak Mau Terbuka
- Pastikan tidak ada proses lain yang sedang menggunakan `/dev/video0`:
  ```bash
  fuser /dev/video0
  ```
- Jika ada proses lain, hentikan dengan `kill -9 <PID>`.

### 2. Inferensi Terlalu Lambat di Raspberry Pi 3
- Ekspor model YOLOv5 ke format ONNX dengan resolusi input lebih kecil (misal: 320x320):
  ```bash
  python export.py --weights best.pt --include onnx --img 320
  ```
- Gunakan varian arsitektur terkecil seperti **YOLOv5n (Nano)**.
- Atur `TORCH_THREADS = 2` pada `config.py` untuk mencegah CPU throttle.

### 3. Timbangan Menunjukkan Nilai Tidak Akurat
- Lakukan kalibrasi nilai `HX711_CALIBRATION_FACTOR` di `config.py` menggunakan objek pembanding dengan berat yang sudah diketahui secara presisi.

### 4. Perubahan UI Tidak Muncul di Browser
- Lakukan hard refresh browser dengan kombinasi tombol `Ctrl + F5` (atau `Cmd + Shift + R` pada Mac) agar cache JavaScript/CSS lama dibersihkan.

---

## 🧪 Menjalankan Unit Test

Untuk memverifikasi fungsi komponen backend:
```bash
python3 -m unittest discover tests
```

---

## 📄 Lisensi & Kontribusi
Dikembangkan untuk sistem analitik pangan cerdas berbasis IoT & Edge AI. Silakan kembangkan dan sesuaikan sesuai kebutuhan penelitian Anda!
