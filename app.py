# ==============================================================================
# FASE 2: BACKEND FLASK - Aplikasi Prediksi Kematian Jawa Barat
# Judul  : Implementasi Backpropagation untuk Prediksi Jumlah Kematian
# Penulis: Muhamad Rafli Ardiansyah
#
# PENTING:
# - Menggunakan tflite_runtime (BUKAN tensorflow) agar ringan saat deploy Vercel
# - Vercel free tier memiliki batas ukuran package < 250MB, tensorflow terlalu besar
# ==============================================================================

import os
import json
import pickle
import numpy as np

from flask import Flask, render_template, request, jsonify

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# ------------------------------------------------------------------------------
# INISIALISASI APLIKASI FLASK
# ------------------------------------------------------------------------------
app = Flask(__name__)

# ------------------------------------------------------------------------------
# LOAD MODEL & PREPROCESSING ARTIFACTS
# Semua file ini dihasilkan dari Fase 1 (fase1_training.py)
# ------------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifacts():
    """
    Memuat semua artifact yang diperlukan untuk prediksi:
    - Model TFLite
    - Label Encoders (Kabupaten, Jenis Kematian, Penyebab)
    - Standard Scaler X dan Y
    - Daftar kelas untuk dropdown
    """
    artifacts = {}

    # Load interpreter TFLite
    model_path = os.path.join(BASE_DIR, 'model.tflite')
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    artifacts['interpreter']     = interpreter
    artifacts['input_details']   = interpreter.get_input_details()
    artifacts['output_details']  = interpreter.get_output_details()

    # Load Label Encoders
    with open(os.path.join(BASE_DIR, 'label_encoders.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    artifacts['le_kabupaten'] = encoders['le_kabupaten']
    artifacts['le_jenis']     = encoders['le_jenis']
    artifacts['le_penyebab']  = encoders['le_penyebab']

    # Load Standard Scaler
    with open(os.path.join(BASE_DIR, 'scaler_X.pkl'), 'rb') as f:
        artifacts['scaler_X'] = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'scaler_y.pkl'), 'rb') as f:
        artifacts['scaler_y'] = pickle.load(f)

    # Load daftar kelas encoding untuk dropdown
    with open(os.path.join(BASE_DIR, 'kelas_encoding.json'), 'r', encoding='utf-8') as f:
        artifacts['kelas'] = json.load(f)

    return artifacts


# Muat artifact saat aplikasi pertama kali dijalankan
print("[INFO] Memuat model dan artifact preprocessing...")
artifacts = load_artifacts()
print("[INFO] Model dan artifact berhasil dimuat!")


# ------------------------------------------------------------------------------
# FUNGSI PREDIKSI
# ------------------------------------------------------------------------------

def prediksi_kematian(nama_kabupaten_kota, jenis_kematian, penyebab_kematian, tahun):
    """
    Melakukan prediksi jumlah kematian berdasarkan input pengguna.

    Args:
        nama_kabupaten_kota (str): Nama kabupaten/kota (sesuai label encoder)
        jenis_kematian      (str): Jenis kematian (sesuai label encoder)
        penyebab_kematian   (str): Penyebab kematian (sesuai label encoder)
        tahun               (int): Tahun data

    Returns:
        dict: Hasil prediksi beserta info tambahan
    """
    try:
        # --- LANGKAH 1: Label Encoding (sama seperti saat training) ---
        kab_encoded     = artifacts['le_kabupaten'].transform([nama_kabupaten_kota])[0]
        jenis_encoded   = artifacts['le_jenis'].transform([jenis_kematian])[0]
        penyebab_encoded= artifacts['le_penyebab'].transform([penyebab_kematian])[0]

        # --- LANGKAH 2: Susun array fitur input ---
        fitur_raw = np.array([[
            float(kab_encoded),
            float(jenis_encoded),
            float(penyebab_encoded),
            float(tahun)
        ]])

        # --- LANGKAH 3: Standard Scaling (sama seperti saat training) ---
        fitur_scaled = artifacts['scaler_X'].transform(fitur_raw).astype(np.float32)

        # --- LANGKAH 4: Inferensi menggunakan TFLite Interpreter ---
        interpreter    = artifacts['interpreter']
        input_details  = artifacts['input_details']
        output_details = artifacts['output_details']

        interpreter.set_tensor(input_details[0]['index'], fitur_scaled)
        interpreter.invoke()
        output_scaled = interpreter.get_tensor(output_details[0]['index'])

        # --- LANGKAH 5: Inverse Transform → kembalikan ke skala asli ---
        prediksi_asli = artifacts['scaler_y'].inverse_transform(output_scaled)[0][0]

        # Pastikan hasil tidak negatif (jumlah kematian tidak bisa negatif)
        prediksi_asli = max(0.0, float(prediksi_asli))

        return {
            'status' : 'success',
            'prediksi': round(prediksi_asli, 2),
            'input'  : {
                'kabupaten_kota' : nama_kabupaten_kota,
                'jenis_kematian' : jenis_kematian,
                'penyebab_kematian': penyebab_kematian,
                'tahun'          : tahun,
            }
        }

    except ValueError as e:
        return {
            'status': 'error',
            'pesan' : f'Nilai input tidak dikenali: {str(e)}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'pesan' : f'Terjadi kesalahan saat prediksi: {str(e)}'
        }


# ------------------------------------------------------------------------------
# ROUTING FLASK
# ------------------------------------------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Route utama:
    - GET  → Tampilkan halaman form input
    - POST → Proses form, lakukan prediksi, kembalikan hasil ke template
    """

    # Ambil daftar pilihan dropdown dari artifact
    kelas       = artifacts['kelas']
    hasil       = None
    form_data   = {}

    if request.method == 'POST':
        # Ambil data dari form HTML
        nama_kabupaten_kota = request.form.get('kabupaten_kota', '').strip()
        jenis_kematian      = request.form.get('jenis_kematian', '').strip()
        penyebab_kematian   = request.form.get('penyebab_kematian', '').strip()
        tahun_str           = request.form.get('tahun', '').strip()

        # Simpan form data untuk ditampilkan kembali (agar tidak reset setelah submit)
        form_data = {
            'kabupaten_kota'   : nama_kabupaten_kota,
            'jenis_kematian'   : jenis_kematian,
            'penyebab_kematian': penyebab_kematian,
            'tahun'            : tahun_str,
        }

        # Validasi input dasar
        if not all([nama_kabupaten_kota, jenis_kematian, penyebab_kematian, tahun_str]):
            hasil = {
                'status': 'error',
                'pesan' : 'Semua field harus diisi!'
            }
        else:
            try:
                tahun = int(tahun_str)
                if tahun < 1900 or tahun > 2100:
                    raise ValueError("Tahun tidak valid")

                # Jalankan prediksi
                hasil = prediksi_kematian(
                    nama_kabupaten_kota,
                    jenis_kematian,
                    penyebab_kematian,
                    tahun
                )

            except ValueError as e:
                hasil = {
                    'status': 'error',
                    'pesan' : f'Tahun harus berupa angka yang valid: {str(e)}'
                }

    return render_template(
        'index.html',
        kelas       = kelas,
        hasil       = hasil,
        form_data   = form_data,
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Endpoint API JSON untuk prediksi (opsional, untuk integrasi eksternal).
    Request body (JSON):
    {
        "kabupaten_kota"   : "KABUPATEN BOGOR",
        "jenis_kematian"   : "KEMATIAN IBU",
        "penyebab_kematian": "PENDARAHAN",
        "tahun"            : 2023
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'status': 'error', 'pesan': 'Body request harus berformat JSON'}), 400

        hasil = prediksi_kematian(
            data.get('kabupaten_kota', ''),
            data.get('jenis_kematian', ''),
            data.get('penyebab_kematian', ''),
            int(data.get('tahun', 0))
        )
        status_code = 200 if hasil['status'] == 'success' else 400
        return jsonify(hasil), status_code

    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint health check untuk memastikan server berjalan."""
    return jsonify({'status': 'ok', 'pesan': 'Server berjalan normal'}), 200


# ------------------------------------------------------------------------------
# JALANKAN SERVER (hanya saat dijalankan langsung, bukan melalui Vercel/WSGI)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5000)
