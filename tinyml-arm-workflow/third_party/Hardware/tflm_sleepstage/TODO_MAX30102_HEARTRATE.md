# TODO - MAX30102 Heart Rate Display (cvapp_mb_cls.cpp)

## Information gathered
- `cvapp_mb_cls.cpp` sudah inisialisasi I2C, MPU6050, dan MAX30102.
- Saat ini MAX30102 hanya menampilkan `HR_RAW` (raw sample 18-bit), belum dihitung menjadi **heart rate (BPM)**.
- Library MAX30102 yang tersedia: `max30102_read_hr_sample()` (HR-only red LED) dan `max30102_available_samples()`.
- Perlu implementasi algoritma sederhana untuk menurunkan HR sample menjadi BPM.

## Plan (step-by-step)
1. Update `cvapp_mb_cls.cpp`
   - Perbaiki struktur pembacaan MAX30102 (hapus shadowing variabel `hr_sample`).
   - Tambahkan buffering beberapa sampel HR.
   - Implementasi deteksi beat sederhana berbasis threshold dan peak finding (tanpa ubah driver C).
   - Hitung BPM = 60 / periode_beat (dalam detik) menggunakan timestamp sampling.
   - Tampilkan `HR_BPM=...` di serial (bukan hanya HR_RAW).
2. Build & test
   - Jalankan build sesuai makefile project.
   - Flash dan verifikasi serial output.

## Done
- [ ] Update algoritma BPM dan output di `cvapp_mb_cls.cpp`
- [ ] Build/test di target board

