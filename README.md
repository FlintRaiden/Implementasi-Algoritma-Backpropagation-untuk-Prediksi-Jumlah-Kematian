# 🏥 Prediksi Jumlah Kematian Jawa Barat — AI Backpropagation

> **Tugas Kuliah Artificial Intelligence**
> Penulis: **Muhamad Rafli Ardiansyah**

Aplikasi web prediksi jumlah kematian di Provinsi Jawa Barat menggunakan
algoritma **Backpropagation (Neural Network)** berbasis TensorFlow/Keras,
dengan deployment Flask + TFLite ke Vercel.

---

## 📁 Struktur Proyek

```
backprop-jabar/
│
├── fase1_training.py          ← FASE 1: Pelatihan model (jalankan sekali)
│
├── app.py                     ← FASE 2: Backend Flask
├── templates/
│   └── index.html             ← FASE 3: Frontend HTML
│
├── requirements.txt           ← Dependensi Python
├── vercel.json                ← Konfigurasi deploy Vercel
│
│   (File yang dihasilkan Fase 1 — tidak ada di repo, generate dulu)
├── model.tflite               ← Model hasil training
├── label_encoders.pkl         ← Encoder kategorikal
├── scaler_X.pkl               ← Scaler fitur input
├── scaler_y.pkl               ← Scaler output
└── kelas_encoding.json        ← Daftar pilihan dropdown
```

---

## 🚀 Cara Menjalankan Proyek

### LANGKAH 1 — Siapkan Lingkungan Python

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan (Linux/Mac)
source venv/bin/activate

# Aktifkan (Windows)
venv\Scripts\activate

# Install dependensi untuk training (butuh tensorflow penuh)
pip install tensorflow pandas numpy scikit-learn matplotlib
```

### LANGKAH 2 — Jalankan Fase 1: Training Model

```bash
# Pastikan file CSV dataset ada di folder yang sama
python fase1_training.py
```

Setelah selesai, akan terbentuk file-file:
- `model.tflite`
- `label_encoders.pkl`
- `scaler_X.pkl`
- `scaler_y.pkl`
- `kelas_encoding.json`
- `grafik_loss_training.png`
- `grafik_aktual_vs_prediksi.png`

### LANGKAH 3 — Jalankan Flask Secara Lokal

```bash
# Install dependensi Flask (gunakan tflite-runtime bukan tensorflow)
pip install Flask tflite-runtime scikit-learn numpy

# Jalankan server
python app.py
```

Buka browser: **http://localhost:5000**

---

## ☁️ Deploy ke Vercel

### Prasyarat
- Akun Vercel gratis di [vercel.com](https://vercel.com)
- Vercel CLI: `npm install -g vercel`

### Langkah Deploy

```bash
# Login ke Vercel
vercel login

# Deploy (dari folder proyek)
vercel

# Deploy ke production
vercel --prod
```

> ⚠️ **Penting:** File `model.tflite`, `*.pkl`, dan `kelas_encoding.json`
> harus sudah ada di folder proyek sebelum deploy, karena file-file
> tersebut tidak bisa di-generate di server Vercel.

---

## 🧠 Detail Arsitektur Model

| Layer         | Neuron | Aktivasi | Keterangan              |
|---------------|--------|----------|-------------------------|
| Input Layer   | 4      | —        | Kab, Jenis, Penyebab, Tahun |
| Hidden Layer 1| 64     | ReLU     | + BatchNorm + Dropout   |
| Hidden Layer 2| 128    | ReLU     | + BatchNorm + Dropout   |
| Hidden Layer 3| 64     | ReLU     | + Dropout               |
| Hidden Layer 4| 32     | ReLU     | —                       |
| Output Layer  | 1      | Linear   | Regresi jumlah kematian |

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** Mean Squared Error (MSE)
- **Regularisasi:** Early Stopping, ReduceLROnPlateau, Dropout, BatchNorm

---

## 📊 Dataset

| Atribut       | Detail                          |
|---------------|---------------------------------|
| Sumber        | Dinas Kesehatan Provinsi Jawa Barat |
| Periode       | 2017 – 2019                     |
| Total Baris   | 2.106 baris                     |
| Fitur Input   | 4 kolom (3 kategorikal, 1 numerik) |
| Target        | `jumlah_kematian` (numerik)     |
| Wilayah       | 27 Kabupaten/Kota               |
| Jenis         | 4 (Balita, Ibu, Post-Neo, Lahir Mati) |
| Penyebab      | 19 penyebab kematian            |

---

## ⚠️ Disclaimer

Aplikasi web ini dibuat khusus untuk memenuhi tugas kuliah matakuliah
Artificial Intelligence oleh **Muhamad Rafli Ardiansyah**.
Hasil prediksi merupakan estimasi statistik dan tidak dapat dijadikan
referensi medis atau kebijakan publik.
