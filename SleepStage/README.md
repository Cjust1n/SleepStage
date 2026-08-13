# SleepStage (Sleep Stage Classification)

Project ini berisi pipeline end-to-end untuk mengklasifikasikan sleep stage menggunakan fitur hasil ekstraksi dari sinyal **accelerometer (movement)** dan **heart rate (HR/HRV)**.

Secara garis besar:

1. **Preprocessing & Feature Extraction** membangun dataset terproses (`processed/`).
2. **Sequence Builder** membentuk input berurutan (time series windows) untuk model.
3. **Subject-wise split** agar evaluasi tidak bocor antar subjek.
4. **Scaling/Normalisasi** fitur.
5. **Training model** (GRU/LSTM, namun implementasi `build_gru_model` didesain “builtins-only” agar konversi quantization INT8 lebih mudah).
6. **Evaluasi** menghasilkan `outputs/metrics.json` + confusion matrix.
7. **Konversi & Quantization INT8 (TFLite)**, lalu **evaluasi perbandingan** FLOAT32 vs INT8.

> Catatan penting: implementasi model `build_gru_model` pada `src/models/rnn.py` menggunakan blok Conv1D dilated causal (TCN-style) sehingga konversi ke TFLite INT8 cenderung lebih stabil dan tetap “builtins-only”.

---

## Struktur folder (apa saja dan ngapain)

### 1) Top-level

- `config.py`
  - Konstanta global preprocessing/epoching (mis. sampling rate, aturan pembersihan epoch, dll.).
  - `SEQUENCE_LENGTH` diset supaya kompatibel untuk kode legacy.

- `configs/baseline.yaml`
  - Konfigurasi hyperparameter training & evaluasi, termasuk:
    - `sequence_length`, `step`, `require_contiguous`
    - `model_type` (`gru` atau `lstm`)
    - ukuran model (hidden units), dropout, optimizer LR, epochs, patience
    - split subject (opsional explicit list, atau auto-partition deterministik)
    - `use_class_weight`
    - quantization INT8: `quantize_int8`, `representative_steps`

- `TODO.md`
  - Catatan pekerjaan lanjutan (terutama PSO untuk class-weight scaling).

- `test_keras.py`
  - Script util sederhana untuk memuat model Keras dan print bentuk input/output.

- `build_all_processed_dataset.py`
  - Entry-point untuk membuat dataset terproses dari semua data mentah.
  - Memanggil parser raw → builder → menghasilkan `processed/X.npy`, `processed/y.npy`, `processed/metadata.csv`.

- `convert_to_cc.py`
  - Kosong/placeholder di snapshot ini (file dibaca tapi isinya tidak ada).

- `export_tflite.py`
  - Script placeholder (terbaca `import tensorflow as tf` dan memuat model).

- `quantization_comparison.json`
  - Output contoh/hasil perbandingan quantization (bisa dibuat ulang).

---

### 2) `Dataset/` (raw dataset + builder dataset)

- `Dataset/raw/`
  - Folder data mentah (Bidslab*). Di repositori ini hanya terlihat struktur folder/dummy index.
  - Dokumentasi/metadata pada `Dataset/raw/README.md`.

- `Dataset/dataset_builder.py`
  - Menyusun dataset akhir dari data yang diparse (night/epoch).

  Alur utama `DatasetBuilder`:

  1. **Align** sinyal motion & HR (`EpochAligner`).
  2. **Segment** menjadi epoch-level (`EpochSegmenter`).
  3. **Cleaning** epoch yang kualitasnya buruk (`EpochCleaner`).
  4. Untuk tiap epoch:
     - Extract fitur HRV dari window HR (`HRVFeatureExtractor`).
     - Extract fitur movement (`MovementFeatureExtractor`).
     - Extract fitur temporal berbasis posisi epoch dalam malam (`TemporalFeatureExtractor`).
     - Gabungkan menjadi vektor fitur: `concat([hrv, motion, temporal])`.
     - Drop epoch yang menghasilkan `NaN/Inf`.
  5. **Remapping label**:
     - Raw label: `0=wake, 1=N1, 2=N2, 3=N3, 4=REM`.
     - Remap menjadi skema 4 kelas contiguous `0..3`:
       - `0 = wake`
       - `1 = N1 & N2`
       - `2 = N3`
       - `3 = REM`
     - Jika ada label selain yang diharapkan, epoch tersebut di-drop (fail-gracefully).

  Output builder:
  - `X.npy` float32: shape `(n_epochs, n_features)`
  - `y.npy` int32: shape `(n_epochs,)`
  - `metadata.csv`: berisi `Subject`, `Night`, `Epoch`

- `Dataset/feature_order.py`
  - Menentukan urutan fitur untuk konsistensi (disimpan sebagai `feature_names.json` di folder `processed/`).

---

### 3) `Preprocessing/` (pipeline preprocessing)

Folder ini berisi komponen yang dipakai oleh `Dataset/dataset_builder.py`:

- `parser.py`
  - Mem-parsing dataset mentah (menyediakan objek night/subject/label).
- `align.py`
  - Menyamakan timeline motion dan HR.
- `epoch_segmenter.py`
  - Memotong sinyal menjadi epoch.
- `cleaning.py`
  - Memfilter epoch berdasarkan kualitas (mis. motion percent, HR percent).
- `sliding_window.py` dan `window_builder.py`
  - Membentuk window untuk ekstraksi fitur.
- `normalizer.py`
  - Normalisasi fitur berdasarkan statistik (mean/std atau sejenisnya).
- `sequence_builder.py`
  - Membuat sequences/time-series windows dari data epoch-level.
- `subject_split.py` (dipakai via wrapper `src/preprocessing/split_subject.py`)
  - Split data berdasarkan subject.

---

### 4) `FeatureExtraction/` (ekstraksi fitur)

Komponen ekstraksi fitur:

- `hrv.py` : fitur HRV
- `movement.py` : fitur movement dari accelerometer/gerakan
- `temporal.py` : fitur temporal (posisi epoch, progress malam, dll.)
- `utils.py` : helper

---

### 5) `processed/` (dataset hasil preprocessing)

Folder ini berisi artefak siap-latih (bisa dianggap sebagai “cache” dataset):

- `X.npy` : fitur ter-normalkan maupun versi mentah (tergantung pipeline). Di repo terlihat `X.npy` dan `X_normalized.npy`.
- `y.npy` : label integer 0..3.
- `metadata.csv` : mapping epoch → subject/night.
- `normalization.json` : parameter normalisasi (jika ada dari pipeline lain).
- `feature_names.json` : urutan nama fitur.

Training pipeline dalam snapshot ini memuat dari `processed/X.npy`, `processed/y.npy`, `processed/metadata.csv`.

---

### 6) `src/` (kode utama training & evaluation)

#### a) `src/datasets/`

- `src/datasets/dataset.py`
  - `load_processed_dataset(processed_dir)`:
    - memuat `X.npy`, `y.npy`, `metadata.csv`.
    - validasi kolom metadata harus punya `Subject`, `Night`, `Epoch`.
  - `infer_num_classes(y)`
    - mengembalikan `max(y)+1`.

#### b) `src/preprocessing/`

- `sequence_builder.py`
  - `SequenceCreator` wrapper untuk `Preprocessing/sequence_builder.py`.

- `scaler.py`
  - `Scaler` wrapper untuk `Preprocessing/normalizer.py`.
  - fit scaler hanya di train set, lalu transform val/test.

- `split_subject.py`
  - `SubjectWiseSplitter` wrapper untuk `Preprocessing/subject_split.py`.

#### c) `src/models/`

- `src/models/lstm.py`
  - Model LSTM classifier:
    - Input `(T, F)`
    - LSTM → Dropout → Dense logits `(num_classes)`
    - Loss: `SparseCategoricalCrossentropy(from_logits=True)`

- `src/models/rnn.py`
  - Fungsi `build_gru_model(...)` (nama tetap “gru” untuk kompatibilitas training),
    namun implementasi internal menggunakan:
    - Stem `Conv1D(filters=hidden_units, kernel_size=1)`
    - Beberapa blok `_TCNBlock` dengan `Conv1D(..., dilation_rate=...)` padding causal
    - Residual connection
    - `GlobalAveragePooling1D` lalu `Dense logits`

  Tujuan desain ini adalah agar model lebih mudah dikonversi ke **TFLite INT8 builtins-only**.

#### d) `src/training/`

- `src/training/train.py` (pipeline utama)

  Alur lengkap training:

  1. Load konfigurasi (`TrainConfig`) dari yaml.
  2. `create_sequences_and_split(cfg)`:
     - load dataset processed
     - buat sequences menggunakan `SequenceCreator(sequence_length, step, require_contiguous)`
     - split subject: train/val/test.
  3. `_scale_sequence_data(...)`:
     - fit scaler pada X_train (flatten time) lalu transform train/val/test.
  4. Build model:
     - jika `model_type == 'gru'` → `build_gru_model` (TCN-style)
     - jika `model_type == 'lstm'` → `build_lstm_model`
     - LR di-set ulang dengan `cfg.learning_rate`.
  5. Class-weight balancing (opsional):
     - compute `compute_class_weight(class_weight='balanced')`
     - kemudian dipangkatkan `exponent_alpha` (default 0.7)
     - dan dikalikan multipliers per-class (default tergantung jumlah kelas yang ada)
     - hasil akhirnya di-clip [0.1, 10.0]

     (Di file ini ada kerangka PSO, tapi tidak semuanya “aktif” dari config default kecuali flag terkait diaktifkan.)

  6. Train model dengan callback:
     - `ModelCheckpoint` save `outputs/checkpoints/best.keras`
     - `EarlyStopping` on `val_loss`
     - `ReduceLROnPlateau` on `val_loss`
  7. Evaluasi:
     - load best checkpoint bila ada
     - `evaluate_model(...)` untuk menghasilkan `outputs/metrics.json` dan confusion matrix
  8. Simpan scaler dan model:
     - `outputs/scaler.json`
     - `outputs/saved_model.keras`
     - export `SavedModel` ke `outputs/saved_model/`
  9. Quantization INT8 (opsional, jika `quantize_int8=true`):
     - panggil `quantize_savedmodel_to_int8(...)`

  Output utama training:
  - `outputs/metrics.json`
  - `outputs/confusion_matrix/*`
  - `outputs/checkpoints/best.keras`
  - `outputs/saved_model.keras` dan `outputs/saved_model/*`
  - `outputs/checkpoints/*_int8.tflite` (jika quantize_int8)
  - `outputs/scaler.json`
  - `outputs/run_summary.json` berisi ringkasan (dan info PSO bila aktif)

- `src/training/evaluate.py`
  - `evaluate_model(model, X_test, y_test)` menghitung metrik:
    - accuracy, balanced accuracy
    - f1_macro, f1_weighted
    - Cohen’s kappa, MCC
    - precision/recall/f1 per kelas
    - confusion matrix + classification_report
  - `save_eval_outputs(...)` menyimpan ke `outputs/metrics.json` dan `outputs/confusion_matrix/*`.

- `src/training/quantize_tflite.py`
  - `quantize_savedmodel_to_int8(saved_model_dir, tflite_path, representative_data, representative_steps)`

  Strategi quantization:
  - Converter: `tf.lite.TFLiteConverter.from_saved_model(...)`
  - Set optimizations: `Optimize.DEFAULT`
  - Tentukan representative dataset (berdasarkan `X_train_s` sequence data float32)
  - Konversi dibuat **fail-fast** builtins-only:
    - `supported_ops = [TFLITE_BUILTINS]`
    - `allow_custom_ops = False`
    - input/output type dipaksa `int8`
  - Ada beberapa “diagnostic passes”:
    - pass builtins_only
    - pass lower_tensor_list_ops (jika ada)
    - pass resource variables enabled (diagnosis)
    - fallback allow SELECT_TF_OPS (hanya diagnosis)

  Tujuan: memastikan model akhirnya benar-benar cocok untuk deploy INT8 builtins-only.

#### e) `src/evaluation/`

- `src/evaluation/evaluate_quantization.py`
  - Membandingkan model Keras FLOAT32 vs TFLite INT8 pada test set yang sama.

  Alur:
  1. Build sequences + split subject yang konsisten.
  2. Scale menggunakan scaler fit pada train (mode evaluasi mengikuti training).
  3. Load model FLOAT32: `tf.keras.models.load_model(..., compile=False)`.
  4. Load interpreter TFLite.
  5. Ambil quantization params dari interpreter:
     - input scale/zero-point
     - output scale/zero-point
  6. Jalankan inferensi INT8:
     - quantize input float32 → int8
     - invoke interpreter per-sample
  7. Dequantize output logits ke float32, lalu hitung metrik yang sama.
  8. Simpan perbandingan ke `quantization_comparison.json` (default) bila sukses.

---

### 7) `Analysis/` dan `Training/README.md`

- `Analysis/*.ipynb`
  - Notebook untuk eksplorasi dataset (statistik, distribusi fitur, korelasi, visualisasi sequence, label distribution, dll.)
  - Output gambar disimpan di `Analysis/figures/`.

- `Training/README.md`
  - Placeholder.

---

### 8) `outputs/`

Folder artefak hasil training & evaluasi:

- `outputs/checkpoints/`
  - `best.keras`
  - `*_int8.tflite` jika quantization dilakukan

- `outputs/confusion_matrix/`
  - `confusion_matrix.csv`
  - `confusion_matrix.png`

- `outputs/metrics.json`
  - metrik evaluasi (FLOAT32 di atas test set)

- `outputs/run_summary.json`
  - ringkasan run (termasuk PSO info bila enabled)

- `outputs/saved_model.keras` dan `outputs/saved_model/`
  - model artefak untuk quantization/deploy

- `outputs/scaler.json`
  - statistik normalisasi untuk konsistensi input

- `outputs/visualization*.ipynb`
  - notebook visualisasi hasil eksperimen

- `outputs/visualization.ipynb` dan `outputs/visualization_comparison.ipynb`
  - (di root juga ada salinan) notebook plot/analisis.

---

## Alur algoritma end-to-end (ringkas)

1. **Raw dataset** → parse (`Preprocessing/parser.py`) dan buang epoch buruk.
2. **Feature extraction** per epoch:
   - HRV (dari window HR)
   - Movement (accel)
   - Temporal (posisi epoch)
3. **Label remap** jadi 4 kelas: wake / N1N2 / N3 / REM.
4. Simpan jadi `processed/X.npy`, `processed/y.npy`, `processed/metadata.csv`.
5. **Sequence building**:
   - bentuk sliding windows of length `sequence_length`
   - step `step`
   - opsional enforce `require_contiguous`
6. **Subject-wise split**:
   - train/val/test berdasarkan Subject.
7. **Scaling** fit on train, transform val/test.
8. **Model**:
   - `model_type=gru` → TCN-style builtins-only sequence classifier
   - `model_type=lstm` → LSTM sequence classifier
   - output logits → sparse cross-entropy
9. **Evaluation** metrics + confusion matrix.
10. **Quantization INT8**:
   - export SavedModel → TFLite INT8 builtins-only (representative dataset)
11. **Evaluate_quantization**:
   - hitung metrik INT8 vs FLOAT32 di test set sama.

---

## Output yang diharapkan

- Setelah building processed dataset:
  - `processed/X.npy`
  - `processed/y.npy`
  - `processed/metadata.csv`
  - `processed/feature_names.json`

- Setelah training:
  - `outputs/metrics.json`
  - `outputs/confusion_matrix/*`
  - `outputs/checkpoints/best.keras`
  - `outputs/saved_model.keras`
  - `outputs/saved_model/*`
  - `outputs/scaler.json`
  - `outputs/checkpoints/*_int8.tflite` (jika quantize_int8)

- Setelah evaluasi quantization:
  - `quantization_comparison.json` (atau file output yang dipilih)

