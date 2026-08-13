# TODO - Sync live_monitoring to 18-feature pipeline

## Objective
Board now emits an 18-feature vector per epoch (matching the retrained `gru_int8_vela.tflite` input `[1, 30, 18]`). Update the `live_monitoring` app to consume/display/log all 18 features in the exact board order.

## 18-feature UART order (from board `feat[]` / `F=` print)
```
0 : relative_position
1 : sd2
2 : sin_time_of_night
3 : rolling_mean_hr
4 : rmssd
5 : time_of_night
6 : energy
7 : rolling_hr_range
8 : acceleration_jerk
9 : rolling_mean_acc
10: rms
11: lf
12: rolling_std_acc
13: zero_crossing
14: hr_slope
15: hf
16: lf_hf
17: hr_delta
```

## Steps
- [x] `models/feature_model.py` — dataclass 16→18, FEATURE_META, from_uart (min 18), get_value, to_dict, clear
- [x] `models/feature_history.py` — NUM_FEATURES 16→18, _get_attr_name order
- [x] `widgets/feature_panel.py` — range(16)→range(18), TREND_FEATURES indices, 6-col grid
- [x] `logger/csv_logger.py` — feature HEADERS to 18 cols, log_feature expects 18
- [x] `test_board_format.py` — FEATURE payload expected 18 values

## Console = picocom
- [x] `gui/main_window.py` `_on_data_received` menampilkan semua baris UART ke `dashboard.console` (sama seperti picocom).
- [x] Hapus `def _on_data_received` duplikat yang ada di bawah `_on_data_received` (dia menimpa yang benar dengan commit yang tidak menampilkan ke console).
- [x] `PREDICTION`/`SCORES` memicu pembaruan `prediction_card` + `confidence_chart`, stage sesuai enum board (0=Wake,1=Light,2=Deep,3=REM).

## START=<unix> (time_of_night dari laptop)
- [x] Firmware `cvapp_mb_cls.cpp`: baca non-blocking `START=<unix>\n` via `read_bytes_nonblock()` untuk set `recording_start_unix` (replace hardcoded).
- [x] Python `_start_recording` mengirim `START=<unix>` ke board memakai `SerialWorker.send_command` (yang menambah `\n`).

## Verification
- [x] Run `python -m py_compile` pada file yang dimodifikasi (semua OK, EXIT=0)
- [ ] Rebuild/flash firmware (`APP_TYPE=tflm_mb_cls`, `xmodem_send.py`) dan cek di picocom `F=` 18 nilai + `SCORES=`/`PREDICTION=` setelah epoch >= 30

