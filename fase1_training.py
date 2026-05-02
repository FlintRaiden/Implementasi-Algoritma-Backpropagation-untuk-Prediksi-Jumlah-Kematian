# ==============================================================================
# FASE 1: DATA PREPROCESSING & PELATIHAN MODEL BACKPROPAGATION
# Judul  : Prediksi Jumlah Kematian di Jawa Barat
# Penulis: Muhamad Rafli Ardiansyah
# Deskripsi: Script ini melakukan preprocessing data, pelatihan model neural
#            network (Backpropagation) menggunakan Keras, evaluasi model, dan
#            mengekspor model ke format TensorFlow Lite (.tflite).
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Gunakan backend non-interaktif agar bisa dijalankan tanpa display
import matplotlib.pyplot as plt
import pickle
import json
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------------------
# LANGKAH 1: LOAD DATASET
# ------------------------------------------------------------------------------
print("=" * 60)
print("[LANGKAH 1] Memuat Dataset...")
print("=" * 60)

# Ganti path sesuai lokasi file CSV Anda
CSV_PATH = "dinkes-od_15944_jml_kematian__jenis_penyebab_kematian_data.csv"

df = pd.read_csv(CSV_PATH)
print(f"Dataset berhasil dimuat. Ukuran: {df.shape[0]} baris x {df.shape[1]} kolom")
print(f"\nKolom yang tersedia: {df.columns.tolist()}")
print(f"\nContoh 5 baris pertama:")
print(df.head())

# Pilih kolom yang relevan saja
df = df[['nama_kabupaten_kota', 'jenis_kematian', 'penyebab_kematian', 'tahun', 'jumlah_kematian']].copy()

# Periksa nilai yang hilang (missing values)
print(f"\nJumlah nilai kosong per kolom:")
print(df.isnull().sum())

# Hapus baris dengan nilai kosong jika ada
df.dropna(inplace=True)
print(f"\nUkuran dataset setelah pembersihan: {df.shape[0]} baris")

# ------------------------------------------------------------------------------
# LANGKAH 2: PREPROCESSING DATA
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 2] Preprocessing Data...")
print("=" * 60)

# --- 2a. Label Encoding untuk kolom kategorikal ---
# Label Encoding mengubah teks menjadi angka (0, 1, 2, ...)
# Lebih efisien daripada One-Hot Encoding untuk kolom dengan banyak kategori

le_kabupaten  = LabelEncoder()
le_jenis      = LabelEncoder()
le_penyebab   = LabelEncoder()

df['kab_encoded']     = le_kabupaten.fit_transform(df['nama_kabupaten_kota'])
df['jenis_encoded']   = le_jenis.fit_transform(df['jenis_kematian'])
df['penyebab_encoded']= le_penyebab.fit_transform(df['penyebab_kematian'])

print("Label Encoding selesai.")
print(f"  - Jumlah kelas Kabupaten/Kota : {len(le_kabupaten.classes_)} kelas")
print(f"  - Jumlah kelas Jenis Kematian : {len(le_jenis.classes_)} kelas")
print(f"  - Jumlah kelas Penyebab       : {len(le_penyebab.classes_)} kelas")

# --- 2b. Persiapkan Fitur (X) dan Target (Y) ---
X = df[['kab_encoded', 'jenis_encoded', 'penyebab_encoded', 'tahun']].values.astype(np.float32)
y = df['jumlah_kematian'].values.astype(np.float32)

print(f"\nShape Fitur (X): {X.shape}")
print(f"Shape Target (y): {y.shape}")
print(f"\nStatistik Target (jumlah_kematian):")
print(f"  Min : {y.min():.2f}")
print(f"  Max : {y.max():.2f}")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std : {y.std():.2f}")

# --- 2c. Standard Scaling untuk normalisasi fitur ---
# Scaling diperlukan agar setiap fitur memiliki skala yang sama,
# sehingga gradient descent bekerja lebih efisien.
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

print("\nStandard Scaling selesai.")
print(f"  Mean fitur setelah scaling: {X_scaled.mean(axis=0).round(4)}")

# ------------------------------------------------------------------------------
# LANGKAH 3: SPLIT DATA (80% Training, 20% Testing)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 3] Split Data Training & Testing...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled,
    test_size=0.2,
    random_state=42
)

print(f"Data Training : {X_train.shape[0]} sampel ({X_train.shape[0]/len(X_scaled)*100:.1f}%)")
print(f"Data Testing  : {X_test.shape[0]} sampel  ({X_test.shape[0]/len(X_scaled)*100:.1f}%)")

# ------------------------------------------------------------------------------
# LANGKAH 4: BANGUN ARSITEKTUR MODEL BACKPROPAGATION (Keras Sequential)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 4] Membangun Arsitektur Model Neural Network...")
print("=" * 60)

# Jumlah fitur input
INPUT_DIM = X_train.shape[1]  # 4 fitur

# Arsitektur:
# - Input Layer  : 4 neuron (kab, jenis, penyebab, tahun)
# - Hidden Layer 1: 64 neuron, aktivasi ReLU + Dropout regularisasi
# - Hidden Layer 2: 128 neuron, aktivasi ReLU + Dropout regularisasi
# - Hidden Layer 3: 64 neuron, aktivasi ReLU
# - Hidden Layer 4: 32 neuron, aktivasi ReLU
# - Output Layer : 1 neuron, aktivasi Linear (regresi nilai kontinu)

model = keras.Sequential([
    # --- Input Layer ---
    layers.Input(shape=(INPUT_DIM,), name='Input_Layer'),

    # --- Hidden Layer 1 ---
    layers.Dense(64, activation='relu', name='Hidden_Layer_1'),
    layers.BatchNormalization(name='BatchNorm_1'),
    layers.Dropout(0.2, name='Dropout_1'),

    # --- Hidden Layer 2 ---
    layers.Dense(128, activation='relu', name='Hidden_Layer_2'),
    layers.BatchNormalization(name='BatchNorm_2'),
    layers.Dropout(0.2, name='Dropout_2'),

    # --- Hidden Layer 3 ---
    layers.Dense(64, activation='relu', name='Hidden_Layer_3'),
    layers.Dropout(0.1, name='Dropout_3'),

    # --- Hidden Layer 4 ---
    layers.Dense(32, activation='relu', name='Hidden_Layer_4'),

    # --- Output Layer (Regresi: aktivasi Linear) ---
    layers.Dense(1, activation='linear', name='Output_Layer')
], name='Backpropagation_Model')

# Tampilkan ringkasan arsitektur model
model.summary()

# ------------------------------------------------------------------------------
# LANGKAH 5: KOMPILASI MODEL
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 5] Mengompilasi Model...")
print("=" * 60)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',       # MSE sebagai fungsi loss regresi
    metrics=['mae']                  # MAE sebagai metrik tambahan
)
print("Model berhasil dikompilasi dengan optimizer='adam', loss='mse'")

# Callback: Early Stopping untuk mencegah overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,            # Berhenti jika val_loss tidak membaik dalam 20 epoch
    restore_best_weights=True,
    verbose=1
)

# Callback: ReduceLROnPlateau untuk mengurangi learning rate secara adaptif
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# ------------------------------------------------------------------------------
# LANGKAH 6: LATIH MODEL
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 6] Melatih Model...")
print("=" * 60)

EPOCHS     = 200    # Maksimum epoch (EarlyStopping dapat menghentikan lebih awal)
BATCH_SIZE = 32     # Jumlah sampel per update gradient

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,   # 15% dari training untuk validasi internal
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

epochs_trained = len(history.history['loss'])
print(f"\nPelatihan selesai! Epoch yang dijalankan: {epochs_trained}")

# ------------------------------------------------------------------------------
# LANGKAH 7: EVALUASI MODEL PADA DATA TEST
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 7] Evaluasi Model pada Data Test...")
print("=" * 60)

# Prediksi pada data test (masih dalam skala normalized)
y_pred_scaled = model.predict(X_test).flatten()

# Kembalikan ke skala asli menggunakan inverse_transform
y_pred_asli = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_test_asli = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Hitung metrik evaluasi
mse  = mean_squared_error(y_test_asli, y_pred_asli)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test_asli, y_pred_asli)
r2   = r2_score(y_test_asli, y_pred_asli)

print(f"┌─────────────────────────────────────────┐")
print(f"│         HASIL EVALUASI MODEL             │")
print(f"├─────────────────────────────────────────┤")
print(f"│  MSE  (Mean Squared Error)  : {mse:>10.4f} │")
print(f"│  RMSE (Root MSE)            : {rmse:>10.4f} │")
print(f"│  MAE  (Mean Absolute Error) : {mae:>10.4f} │")
print(f"│  R²   (R-Squared)           : {r2:>10.4f} │")
print(f"└─────────────────────────────────────────┘")

# Tampilkan beberapa contoh prediksi vs aktual
print("\nContoh Prediksi vs Nilai Aktual (10 sampel pertama):")
print(f"{'No':>3} | {'Aktual':>10} | {'Prediksi':>10} | {'Selisih':>10}")
print("-" * 42)
for i in range(min(10, len(y_test_asli))):
    selisih = abs(y_test_asli[i] - y_pred_asli[i])
    print(f"{i+1:>3} | {y_test_asli[i]:>10.2f} | {y_pred_asli[i]:>10.2f} | {selisih:>10.2f}")

# ------------------------------------------------------------------------------
# LANGKAH 8: VISUALISASI GRAFIK LOSS
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 8] Membuat Visualisasi Grafik Loss...")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Hasil Pelatihan Model Backpropagation\nPrediksi Jumlah Kematian - Jawa Barat',
             fontsize=13, fontweight='bold', y=1.02)

# --- Plot 1: Loss (MSE) per Epoch ---
ax1 = axes[0]
ax1.plot(history.history['loss'],     label='Training Loss',   color='#DC2626', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', color='#1D4ED8', linewidth=2, linestyle='--')
ax1.set_title('Grafik Loss (MSE) per Epoch', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Mean Squared Error (MSE)', fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)
ax1.set_facecolor('#F9FAFB')

# Tandai epoch terbaik
best_epoch = np.argmin(history.history['val_loss'])
ax1.axvline(x=best_epoch, color='green', linestyle=':', alpha=0.7, label=f'Best Epoch: {best_epoch+1}')
ax1.legend(fontsize=10)

# --- Plot 2: MAE per Epoch ---
ax2 = axes[1]
ax2.plot(history.history['mae'],     label='Training MAE',   color='#DC2626', linewidth=2)
ax2.plot(history.history['val_mae'], label='Validation MAE', color='#1D4ED8', linewidth=2, linestyle='--')
ax2.set_title('Grafik MAE (Mean Absolute Error) per Epoch', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.4)
ax2.set_facecolor('#F9FAFB')

plt.tight_layout()
plt.savefig('grafik_loss_training.png', dpi=150, bbox_inches='tight')
print("Grafik loss berhasil disimpan ke: grafik_loss_training.png")
plt.close()

# --- Plot tambahan: Aktual vs Prediksi ---
fig2, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(y_test_asli, y_pred_asli, alpha=0.5, color='#DC2626', s=40, edgecolors='white', linewidth=0.5)
max_val = max(y_test_asli.max(), y_pred_asli.max())
ax3.plot([0, max_val], [0, max_val], 'b--', linewidth=2, label='Prediksi Sempurna (y=x)')
ax3.set_title(f'Aktual vs Prediksi (R² = {r2:.4f})', fontsize=12, fontweight='bold')
ax3.set_xlabel('Jumlah Kematian Aktual', fontsize=11)
ax3.set_ylabel('Jumlah Kematian Prediksi', fontsize=11)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.4)
ax3.set_facecolor('#F9FAFB')
plt.tight_layout()
plt.savefig('grafik_aktual_vs_prediksi.png', dpi=150, bbox_inches='tight')
print("Grafik aktual vs prediksi berhasil disimpan ke: grafik_aktual_vs_prediksi.png")
plt.close()

# ------------------------------------------------------------------------------
# LANGKAH 9: SIMPAN MODEL KE FORMAT TENSORFLOW LITE (.tflite)
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 9] Mengonversi & Menyimpan Model ke TFLite...")
print("=" * 60)

# Konversi model Keras ke TFLite menggunakan TFLiteConverter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optimasi ukuran model (quantization ringan)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Lakukan konversi
tflite_model = converter.convert()

# Simpan file .tflite
TFLITE_PATH = 'model.tflite'
with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

ukuran_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"Model TFLite berhasil disimpan ke: {TFLITE_PATH}")
print(f"Ukuran file model: {ukuran_kb:.2f} KB ({ukuran_kb/1024:.2f} MB)")

# ------------------------------------------------------------------------------
# LANGKAH 10: SIMPAN ENCODER DAN SCALER
# Agar saat prediksi di Flask, encoding/scaling-nya IDENTIK dengan saat training
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[LANGKAH 10] Menyimpan Encoder & Scaler...")
print("=" * 60)

# Simpan semua encoder dan scaler ke file pickle
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({
        'le_kabupaten': le_kabupaten,
        'le_jenis'    : le_jenis,
        'le_penyebab' : le_penyebab,
    }, f)

with open('scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)

with open('scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

print("Label Encoders disimpan ke: label_encoders.pkl")
print("Scaler X disimpan ke: scaler_X.pkl")
print("Scaler Y disimpan ke: scaler_y.pkl")

# Simpan juga daftar kelas encoding ke JSON (untuk dropdown di frontend)
kelas_dict = {
    'nama_kabupaten_kota': le_kabupaten.classes_.tolist(),
    'jenis_kematian'     : le_jenis.classes_.tolist(),
    'penyebab_kematian'  : le_penyebab.classes_.tolist(),
    'tahun_min'          : int(df['tahun'].min()),
    'tahun_max'          : int(df['tahun'].max()),
}

with open('kelas_encoding.json', 'w', encoding='utf-8') as f:
    json.dump(kelas_dict, f, ensure_ascii=False, indent=2)

print("Daftar kelas encoding disimpan ke: kelas_encoding.json")

# ------------------------------------------------------------------------------
# VERIFIKASI: Uji Interpreter TFLite
# ------------------------------------------------------------------------------
print("\n" + "=" * 60)
print("[VERIFIKASI] Menguji model TFLite dengan 1 sampel prediksi...")
print("=" * 60)

# Load interpreter TFLite (simulasi seperti di Flask nanti)
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ambil satu sampel dari test set untuk verifikasi
sampel_input = X_test[0:1].astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], sampel_input)
interpreter.invoke()
output_tflite = interpreter.get_tensor(output_details[0]['index'])

prediksi_tflite = scaler_y.inverse_transform(output_tflite)[0][0]
aktual_nilai    = y_test_asli[0]

print(f"Nilai Aktual   : {aktual_nilai:.2f}")
print(f"Prediksi TFLite: {prediksi_tflite:.2f}")
print(f"Selisih        : {abs(aktual_nilai - prediksi_tflite):.2f}")

print("\n" + "=" * 60)
print("✅ SEMUA LANGKAH FASE 1 BERHASIL DISELESAIKAN!")
print("=" * 60)
print("\nFile yang dihasilkan:")
print("  📦 model.tflite         → Model utama untuk Flask")
print("  🗂️  label_encoders.pkl   → Encoder kategorikal")
print("  📏 scaler_X.pkl         → Scaler untuk fitur input")
print("  📏 scaler_y.pkl         → Scaler untuk output prediksi")
print("  🗝️  kelas_encoding.json  → Daftar pilihan dropdown")
print("  📊 grafik_loss_training.png")
print("  📊 grafik_aktual_vs_prediksi.png")
