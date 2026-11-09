# Product Requirements Document (PRD): Sistem Visi Komputer Deteksi Objek Real-Time

## 1. Konsep Dasar (Executive Summary)

Produk ini adalah sistem visi komputer berbasis *Artificial Intelligence* (AI) yang mampu menerima *feed* video dari kamera secara *real-time*, kemudian mendeteksi, melokalisasi, dan mengidentifikasi objek di dalam *frame* tersebut. Sistem ini dirancang sebagai aplikasi *edge computing* yang berjalan secara lokal untuk meminimalisasi latensi dan menjaga privasi data, menggunakan arsitektur *Single-Stage Detector* untuk mencapai pemrosesan berkecepatan tinggi.

## 2. Tujuan dan Metrik Kesuksesan (Goals & Success Metrics)

* **Tujuan Utama:** Membangun *Minimum Viable Product* (MVP) pipeline deteksi objek yang stabil, efisien, dan dapat dijalankan di lingkungan lokal tanpa bergantung pada API *cloud* eksternal.
* **Target Latensi:** Mempertahankan kecepatan *inference* minimal 30 FPS (Frames Per Second) pada perangkat keras standar.
* **Target Akurasi:** Mencapai tingkat kepercayaan (*confidence score*) di atas 75% untuk klasifikasi objek umum pada kondisi pencahayaan normal.

---

## 3. Alur Kerja Sistem (System Workflow)

Sistem beroperasi melalui *pipeline* linier yang memproses data dari perangkat keras hingga menghasilkan *output* visual:

1. **Data Ingestion (Input):** Modul kamera menangkap gambar mentah (*raw frame*) secara kontinu.
2. **Preprocessing:** Sistem mengonversi *frame* ke dalam format matriks piksel, melakukan penyesuaian resolusi (misal: *resize* ke 640x640), dan normalisasi warna agar sesuai dengan standar *input* model AI.
3. **Inference (Pemrosesan Inti):** *Frame* yang sudah diproses diumpankan ke dalam model *neural network* (YOLO). Model mengekstraksi fitur, membagi gambar ke dalam *grid*, dan memprediksi kotak koordinat (*bounding box*) serta label kelas secara serentak.
4. **Post-processing (NMS):** Sistem menerapkan algoritma *Non-Maximum Suppression* untuk membersihkan prediksi yang tumpang tindih pada satu objek yang sama, menyisakan satu *bounding box* dengan skor probabilitas tertinggi.
5. **Output Generation:** Sistem menggambar *bounding box*, label nama, dan persentase keyakinan di atas *frame* asli, lalu menampilkannya kembali ke antarmuka pengguna (GUI) atau menyimpannya sebagai *log* data.

---

## 4. Kebutuhan Sistem (System Requirements)

### A. Kebutuhan Perangkat Keras (Hardware)

| Komponen | Spesifikasi Minimum | Catatan |
| --- | --- | --- |
| **Kamera** | Webcam 720p (30 FPS) | Bisa menggunakan kamera laptop bawaan atau USB eksternal. |
| **Komputasi** | CPU Multi-core modern (opsional: GPU diskrit) | Eksekusi berbasis CPU dimungkinkan untuk model nano/ringan. |
| **RAM** | 8 GB | Diperlukan untuk memuat bobot model (*model weights*) ke memori. |

### B. Kebutuhan Perangkat Lunak & Tech Stack (Software)

| Lapisan | Teknologi yang Digunakan | Fungsi |
| --- | --- | --- |
| **Bahasa Pemrograman** | Python / C++ | Pembuatan *script* integrasi, manajemen logika, dan manipulasi data tingkat rendah. |
| **Computer Vision** | OpenCV | Mengakses *hardware* kamera, membaca *frame* per detik, dan menggambar anotasi grafis (*bounding box*). |
| **AI Framework** | PyTorch & Model YOLO (mis. YOLOv8 Nano) | Mesin *inference* utama yang menyimpan arsitektur jaringan saraf dan bobot terlatih. |
| **Environment** | WSL (Windows Subsystem for Linux) / Debian | Lingkungan sistem operasi berbasis Linux untuk stabilitas kompilasi pustaka AI dan eksekusi jaringan. |
| **Deployment** | Docker | Kontainerisasi seluruh *pipeline* (dependensi, OS dasar, dan kode program) agar siap di-*deploy* di server atau perangkat lain tanpa konflik versi. |

---

## 5. Fitur MVP (Minimum Viable Product)

* **Live Stream Detection:** Menampilkan jendela *feed* kamera langsung dengan penanda objek yang bergerak secara *real-time*.
* **Dynamic Object Filtering:** Parameter yang memungkinkan pengguna memilih hanya objek tertentu yang dideteksi (misalnya: hanya deteksi "Manusia" dan "Mobil", abaikan yang lain).
* **Threshold Adjustment:** Variabel yang bisa diubah untuk mengatur batas minimal *confidence score* (misal: abaikan deteksi jika AI tidak yakin di atas 50%).
* **Containerized Execution:** Menyediakan `Dockerfile` atau `docker-compose.yml` agar sistem bisa di- *build* dan di-*run* dengan satu perintah terminal.

## 6. Kriteria Penerimaan (Acceptance Criteria)

1. Aplikasi berhasil mengakses kamera perangkat melalui OpenCV tanpa *crash*.
2. Model berhasil dimuat (di-*load*) ke memori tanpa *out-of-memory error*.
3. Aplikasi mampu mengenali minimal 3 jenis objek dasar (misal: orang, ponsel, botol) saat ditunjukkan ke depan kamera.
4. *Bounding box* tidak berkedip (*flicker*) secara ekstrem atau menghasilkan puluhan kotak tumpang tindih pada satu objek (NMS berfungsi dengan baik).
5. *Image build* menggunakan Docker berhasil dijalankan di lingkungan Linux/WSL tanpa *error library* yang hilang (seperti `libGL.so`).

## 7.struktur file
object-detection-app/
├── .gitignore               # Daftar file/folder yang tidak di-push ke Git (misal: model AI, __pycache__)
├── README.md                # Dokumentasi cara run aplikasi dan build Docker
├── requirements.txt         # Daftar library Python (opencv-python, ultralytics, dll)
├── Dockerfile               # Skrip untuk membangun container/environment aplikasi
├── config/
│   └── settings.yaml        # File konfigurasi (setting threshold NMS, resolusi kamera, target kelas)
├── models/
│   └── yolov8n.pt           # Tempat menyimpan file bobot (weights) AI YOLO (biasanya diunduh terpisah)
└── src/
    ├── __init__.py          # Menandakan folder src sebagai modul Python
    ├── main.py              # Entry point: Mengakses kamera, looping frame, dan menampilkan GUI
    ├── detector.py          # Logika inti AI: Class untuk load model YOLO dan proses inference
    └── utils.py             # Fungsi bantuan: Menggambar bounding box, kalkulasi FPS, dll