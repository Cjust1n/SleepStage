# TODO — Fix Input Shape Mismatch [1,30,18] vs [1,30,16]

## Goal
Make board feature pipeline produce the 18 features (in the exact config order) that the
flashed TFLite model expects, fixing the boot-time "Unexpected input shape" error.

## Steps
- [x] Investigate root cause (model expects 18 features, board builds 16)
- [ ] Edit `movement_features.h` — add `rms` and `acceleration_jerk` fields
- [ ] Edit `movement_features.c` — compute `rms` and `acceleration_jerk`
- [ ] Edit `cvapp_mb_cls.cpp`:
  - [ ] Change `NUM_FEATURES` 16 -> 18
  - [ ] Update `EXPECTED_INPUT_BYTES` 480 -> 540
  - [ ] Add HR rolling-history state (trend window 5)
  - [ ] Add accel mean-acc history state (keep 3)
  - [ ] Add `sin_time_of_night`
  - [ ] Add rolling feature helpers (rolling_mean_hr, rolling_hr_range, hr_slope, hr_delta, rolling_mean_acc, rolling_std_acc)
  - [ ] Rebuild 18-feature vector in exact config order
  - [ ] Update `F=` UART print to match the 18-feature order
- [ ] Rebuild firmware + re-flash

