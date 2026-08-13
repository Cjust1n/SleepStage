/*
 * peak_detector_improved.h
 *
 * Improved threshold-based PPG peak detector.
 *
 * Improvements over the original peak_detector.h:
 *
 *   Parameter          Old    New    Reason
 *   -----------------  -----  -----  -------------------------------
 *   MA_WINDOW          3      5      200ms window smoother @25Hz
 *   THRESHOLD_RATIO    0.30   0.15   Catches smaller-amplitude beats
 *   REFRACTORY_MS      300    250    Prevents missing valid peaks
 *   PROMINENCE_RATIO   n/a    0.10   Filters motion-artifact spikes
 *   Signal quality     n/a    yes    Discards noisy epochs
 *
 * Pipeline:
 *   1. Moving average (MA-5) — reduces high-frequency noise
 *   2. Adaptive threshold — 15% of amplitude (was 30%)
 *   3. Local maxima > threshold — candidate peaks
 *   4. Prominence check — reject spikes without a trough
 *   5. Refractory period — 250 ms minimum inter-peak gap
 *   6. Physiological bounds — RR [250, 1200] ms ([50, 240] BPM)
 *   7. Signal quality index — output for downstream decision
 */

#ifndef PEAK_DETECTOR_IMPROVED_H
#define PEAK_DETECTOR_IMPROVED_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * Configuration Parameters
 * ===========================================================================*/

/* Maximum peaks per epoch window */
#define PEAK_IMP_MAX                20

/* Moving average window (samples). 5 @ 25 Hz = 200 ms. */
#define MA_WINDOW_IMP               5

/* Threshold ratio as fraction of amplitude. 0.15 = 15 %. */
#define THRESHOLD_RATIO_IMP         0.15f

/* Refractory period (ms). Minimum gap between two peaks. */
#define REFRACTORY_MS_IMP           250.0f

/* Prominence: a peak must rise at least this fraction of threshold
 * above the preceding trough. Filters out motion spikes. */
#define PROMINENCE_RATIO_IMP        0.10f

/* Physiological RR interval limits (ms) */
#define RR_MIN_MS_IMP               250   /* 240 BPM max */
#define RR_MAX_MS_IMP               1200  /*  50 BPM min */

/* Minimum signal quality index to accept an epoch (0.0 - 1.0) */
#define SIGNAL_QUALITY_MIN          0.30f

/* =============================================================================
 * Data Structures
 * ===========================================================================*/

typedef struct {
    uint16_t peak_indices[PEAK_IMP_MAX];  /* Sample indices of each peak */
    uint16_t peak_count;                   /* Number of detected peaks */
    float    rr_ms[PEAK_IMP_MAX];          /* RR intervals in ms */
    uint16_t rr_count;                     /* Number of valid RR intervals */
    float    threshold;                    /* Adaptive threshold used */
    float    max_amp;                      /* Max amplitude in window */
    float    min_amp;                      /* Min amplitude in window */
    float    amplitude;                    /* Peak-to-peak amplitude */
    float    signal_quality;               /* 0.0 (bad) .. 1.0 (perfect) */
} peak_improved_result_t;

/* =============================================================================
 * Public API
 * ===========================================================================*/

/**
 * @brief Run the improved peak detector pipeline on one PPG epoch.
 *
 * Steps:
 *   1. MA-5 moving average
 *   2. Adaptive threshold (15 % of amplitude)
 *   3. Local maxima detection with prominence check
 *   4. Refractory period enforcement (250 ms)
 *   5. RR interval computation with physiological bounds
 *   6. Signal quality index
 *
 * @param ppg         PPG signal (float, already high-pass filtered)
 * @param length      Number of samples in the window
 * @param fs          Sample rate in Hz (e.g. 25.0)
 * @param result      Output structure (must be non-NULL)
 */
void peak_detector_improved_process(
    const float *ppg,
    uint16_t length,
    float fs,
    peak_improved_result_t *result);

/**
 * @brief Compute a signal quality index from the detected peaks.
 *
 * The index measures the regularity of the RR intervals.
 * A perfectly regular rhythm (e.g. pacemaker) scores 1.0.
 * Highly irregular or noisy signals score near 0.0.
 *
 * Calculation:
 *   quality = 1.0 - (std(RR) / mean(RR))
 *   clamped to [0.0, 1.0]
 *
 * @param ppg         PPG signal (unused, reserved)
 * @param length      Signal length (unused, reserved)
 * @param peaks       Array of peak sample indices
 * @param peak_count  Number of peaks
 * @return float      Quality index in [0.0, 1.0]
 */
float peak_detector_quality_index(
    const float *ppg,
    uint16_t length,
    const uint16_t *peaks,
    uint16_t peak_count);

/**
 * @brief Get mean heart rate from a result structure.
 *
 * @param result  Pointer to result (must be non-NULL)
 * @return float  Mean HR in BPM, or 0.0 if no valid RR intervals
 */
float peak_detector_improved_get_hr(
    const peak_improved_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* PEAK_DETECTOR_IMPROVED_H */

