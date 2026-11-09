# Object Detection System - Real-time Computer Vision

Sistem visi komputer berbasis AI untuk deteksi objek real-time menggunakan **YOLOv8** (Ultralytics). Dirancang sebagai aplikasi *edge computing* yang berjalan lokal tanpa ketergantungan cloud.

## 🎯 Fitur Utama

- **Live Stream Detection**: Deteksi objek real-time dari feed kamera
- **Dynamic Object Filtering**: Pilih kelas objek yang ingin dideteksi (via config)
- **Threshold Adjustment**: Atur confidence & IoU threshold
- **Containerized**: Siap deploy dengan Docker & Docker Compose
- **Cross-platform**: Berjalan di Linux, WSL2, Windows (via WSL)

## 📋 Persyaratan Sistem

| Komponen | Minimum | Direkomendasikan |
|----------|---------|------------------|
| **OS** | Windows 10/11 (WSL2), Ubuntu 20.04+ | Ubuntu 22.04 / WSL2 Ubuntu |
| **CPU** | Modern multi-core (x64/ARM64) | Intel i5/AMD Ryzen 5 atau lebih baik |
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | Tidak wajib (CPU mode) | NVIDIA GPU + CUDA untuk performa maksimal |
| **Kamera** | 720p @ 30 FPS | 1080p @ 30/60 FPS |
| **Docker** | 20.10+ | Latest stable |
| **Docker Compose** | 2.0+ | Latest stable |

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone <repository-url>
cd object-detection-app
```

### 2. Build & Run dengan Docker (Recommended)

```bash
# Build image
docker compose build

# Run dengan akses kamera (Linux/WSL2)
docker compose up object-detection

# Atau mode development (hot reload)
docker compose up dev
# Lalu di terminal terpisah:
docker compose exec dev python src/main.py
```

### 3. Run Lokal (Tanpa Docker)

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/WSL/Mac
# venv\Scripts\activate   # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run
python src/main.py
```

## ⚙️ Konfigurasi

Edit file `config/settings.yaml`:

```yaml
# Camera
camera:
  device_index: 0        # 0 = default camera
  width: 640
  height: 480
  fps: 30

# Model (YOLOv8 variants: n, s, m, l, x)
model:
  path: "models/yolov8n.pt"

# Inference thresholds
inference:
  conf_threshold: 0.5    # Confidence minimum
  iou_threshold: 0.45    # NMS IoU threshold
  device: "cpu"          # "cpu", "0" (GPU), "mps" (Apple Silicon)

# Filter kelas yang dideteksi (kosong = semua 80 kelas COCO)
filter:
  enabled: true
  target_classes:
    - "person"
    - "cell phone"
    - "bottle"
```

### Daftar Kelas COCO Tersedia

```bash
python src/main.py --list-classes
```

Beberapa kelas umum: `person`, `bicycle`, `car`, `motorcycle`, `bus`, `truck`, `cell phone`, `laptop`, `book`, `bottle`, `cup`, `chair`, `dog`, `cat`, dll.

## 📁 Struktur Project

```
object-detection-app/
├── .gitignore
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── config/
│   └── settings.yaml
├── models/
│   └── yolov8n.pt          # Auto-download saat pertama run
└── src/
    ├── __init__.py
    ├── main.py             # Entry point
    ├── detector.py         # Core AI logic (YOLO wrapper)
    └── utils.py            # Helpers: draw, FPS, config, camera
```

## 🎮 Kontrol Runtime

| Tombol | Aksi |
|--------|------|
| `q` / `ESC` | Keluar aplikasi |
| `s` | Simpan screenshot frame saat ini |

## 🐳 Docker Details

### GPU Support (NVIDIA)

Uncomment bagian `deploy` di `docker-compose.yml` dan pastikan:
1. NVIDIA Driver terinstall di host
2. `nvidia-container-toolkit` terinstall
3. Docker daemon dikonfigurasi untuk GPU

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Camera Access di WSL2

Pastikan:
1. WSL2 dengan kernel terbaru (`wsl --update`)
2. Kamera tidak digunakan aplikasi Windows lain
3. Device `/dev/video0` tersedia di WSL

```bash
# Check camera di WSL
ls -la /dev/video*
v4l2-ctl --list-devices
```

## 🔧 Development

### Menambah Dependensi

```bash
# Update requirements.txt
pip freeze > requirements.txt

# Rebuild Docker
docker compose build --no-cache
```

### Model Custom

Letakkan file `.pt` custom di folder `models/` dan update `config/settings.yaml`:

```yaml
model:
  path: "models/custom_model.pt"
```

### Export ONNX (Untuk Deploy Lebih Cepat)

```python
from ultralytics import YOLO
model = YOLO("models/yolov8n.pt")
model.export(format="onnx", imgsz=640)
```

## 📊 Performa Target

| Metrik | Target |
|--------|--------|
| **FPS (CPU)** | 15-30 FPS (YOLOv8n) |
| **FPS (GPU)** | 60+ FPS (YOLOv8n) |
| **Latency** | < 50ms per frame |
| **Accuracy** | > 75% confidence (normal lighting) |
| **Memory** | ~500MB (YOLOv8n) |

## 🐛 Troubleshooting

### `libGL.so.1: cannot open shared object file`
```bash
# Sudah dihandle di Dockerfile, tapi jika lokal:
sudo apt-get install libgl1-mesa-glx
```

### Kamera tidak terbuka
```bash
# Cek device
ls /dev/video*
# Cek permission
sudo usermod -aG video $USER
# Logout & login kembali
```

### Model tidak ter-download
```bash
# Manual download
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
```

### FPS rendah di CPU
- Gunakan model `yolov8n.pt` (nano) - paling ringan
- Kurangi `imgsz` di config (misal: 320)
- Aktifkan `half_precision: true` (butuh GPU)

## 📝 Lisensi

MIT License - Bebas digunakan untuk keperluan komersial maupun non-komersial.

## 🤝 Kontribusi

1. Fork repository
2. Buat feature branch (`git checkout -b feature/nama-fitur`)
3. Commit changes (`git commit -am 'Add fitur baru'`)
4. Push ke branch (`git push origin feature/nama-fitur`)
5. Buat Pull Request

## 📞 Dukungan

- Buat **Issue** untuk bug report atau feature request
- Cek **Discussions** untuk pertanyaan umum