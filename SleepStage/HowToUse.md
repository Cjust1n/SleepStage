# HowToUse

Dokumen ini berisi instruksi penggunaan kode untuk:
1) membangun dataset terproses (`processed/`)
2) training model
3) quantization INT8 ke TFLite
4) evaluasi perbandingan FLOAT32 vs INT8

> Semua perintah dijalankan dari root project: `/home/christopher-justin/Downloads/SleepStage`.

---

## 0) Persiapan

- Pastikan folder data mentah tersedia di `Dataset/raw/` sesuai struktur yang digunakan `Preprocessing/parser.py`.
- Siapkan environment Python yang sudah memiliki dependency project (TensorFlow, numpy, pandas, scikit-learn, dll.).

---

## 1) Build dataset terproses (`processed/`)

Script:
- `build_all_processed_dataset.py`

Perintah:
```bash
python build_all_processed_dataset.py
```

Output yang akan dihasilkan:
- `processed/X.npy`
- `processed/y.npy`
- `processed/metadata.csv`
- `processed/feature_names.json`

Folder `processed/` digunakan oleh training & evaluasi.

---

## 2) Training model (FLOAT32)

Script utama:
- `src/training/train.py`

Contoh config:
- `configs/baseline.yaml`

Perintah:
```bash
python -m src.training.train --config configs/baseline.yaml
```

Parameter penting pada `configs/baseline.yaml`:
- `sequence_length`
- `step`
- `require_contiguous`
- `model_type` (`gru` atau `lstm`)
- `use_class_weight` (balancing kelas; bisa dipakai bareng `use_undersampling`)
- `use_undersampling` (reduce training data tanpa mematikan weighted loss)
- `quantize_int8` (jika true, akan mengekspor INT8 TFLite setelah training)
- `representative_steps` (jumlah sample untuk kalibrasi quantization)

Output training biasanya berada di:
- `outputs/metrics.json`
- `outputs/confusion_matrix/`
- `outputs/checkpoints/best.keras`
- `outputs/saved_model.keras`
- `outputs/saved_model/`
- `outputs/scaler.json`
- `outputs/checkpoints/*_int8.tflite` (jika `quantize_int8=true`)

---

## 3) Evaluasi Quantization INT8 vs FLOAT32

Script:
- `src/evaluation/evaluate_quantization.py`

Perintah contoh:
```bash
python -m src.evaluation.evaluate_quantization \
  --keras_model outputs/saved_model.keras \
  --tflite_model outputs/checkpoints/gru_int8.tflite \
  --processed_dir processed \
  --sequence_length 30 \
  --step 1 \
  --require_contiguous 1 \
  --output_json quantization_comparison.json
```

Parameter yang harus dicocokkan dengan training:
- `--sequence_length`
- `--step`
- `--require_contiguous`

Jika ingin subject split persis seperti yang diinginkan, gunakan:
- `--val_subjects subject1,subject2`
- `--test_subjects subjectX`

Catatan:
- Jika `--val_subjects` dan `--test_subjects` tidak diberikan, script melakukan partition deterministik berdasarkan urutan subject.

Output:
- metrik FLOAT32 & INT8, lalu saved ke `--output_json`.

---

## 4) (Opsional) Ekspor model & inspeksi input/output

Ada beberapa script util/placeholder seperti `export_tflite.py` dan `test_keras.py`.

Contoh:
```bash
python test_keras.py
```

Tujuan utamanya hanya memastikan input/output shape model.

---

## 5) Troubleshooting ringkas

### A) INT8 TFLite conversion gagal
Saran:
- Pastikan `representative_data` yang dipakai `quantize_tflite.py` sesuai pipeline (di training kode sudah memakai `X_train_s`, bukan `processed/X_normalized.npy`).
- Model `model_type=gru` sebenarnya menggunakan TCN-style builtins-only. Jika memakai LSTM, conversion bisa lebih kompleks.
- Lihat error traceback dari `src/training/quantize_tflite.py` (konversi dibuat fail-fast builtins-only).

### B) Mismatch sequence_length saat evaluasi quantization
- Pastikan `--sequence_length` pada `evaluate_quantization.py` sama dengan training.

---

## Referensi file yang relevan

- Building processed dataset: `build_all_processed_dataset.py`
- Dataset builder: `Dataset/dataset_builder.py`
- Training: `src/training/train.py`
- Model: `src/models/rnn.py` dan `src/models/lstm.py`
- Evaluasi FLOAT32: `src/training/evaluate.py`
- Quantization: `src/training/quantize_tflite.py`
- Evaluasi quantization: `src/evaluation/evaluate_quantization.py`
