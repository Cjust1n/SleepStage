/* =============================================================================
 * peak_detector_threshold.h  --  Simple threshold-based PPG peak detector
 * =============================================================================
 *
 * ALGORITHM: 
 *   1. Apply MA-5 (moving average) to reduce noise
 *   2. Compute adaptive threshold = min + ratio * (max - min)
 *   3. Find local maxima above threshold
 *   4. Apply refractory period to avoid double detection
 *
 * This is a simpler alternative to the Elgendi-style detector, optimized
 * for noisy PPG signals and embedded systems (Cortex-M55).
 * ===========================================================================
 */

#ifndef PEAK_DETECTOR_THRESHOLD_H
#define PEAK_DETECTOR_THRESHOLD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * CONFIGURABLE PARAMETERS
 * ===========================================================================*/

/* Moving average window (reduces noise) */
#define MA_WINDOW               10

/* Threshold ratio: peak must be > min + ratio * (max - min) */
#define THRESHOLD_RATIO         0.90f

/* Refractory period (minimum gap between peaks) in milliseconds */
#define REFRACTORY_MS           400.0f

/* Physiological RR bounds (same as original peak detector) */
#define RR_MIN_MS               350.0f
#define RR_MAX_MS               2000.0f

/* Maximum number of peaks per window */
#define PEAK_MAX                 75

/* =============================================================================
 * DATA STRUCTURES
 * ===========================================================================*/

typedef struct {
    uint16_t peak_indices[PEAK_MAX];   /* Sample indices of detected peaks */
    uint16_t peak_count;                /* Number of detected peaks */
    float rr_ms[PEAK_MAX];              /* RR intervals in milliseconds */
    uint16_t rr_count;                  /* Number of valid RR intervals */
    float threshold;                    /* Adaptive threshold used */
    float max_amp;                      /* Maximum amplitude in window */
    float min_amp;                      /* Minimum amplitude in window */
} peak_threshold_result_t;

/* =============================================================================
 * API FUNCTIONS
 * ===========================================================================*/

/**
 * @brief Initialize peak detector (reset state)
 */
void peak_detector_threshold_init(void);

/**
 * @brief Process a window of PPG data and detect peaks
 * 
 * @param ppg         Pointer to PPG samples (float, normalized)
 * @param length      Number of samples in window
 * @param fs          Sample rate in Hz (e.g., 50.0)
 * @param result      Pointer to result structure (output)
 * 
 * @note The input PPG should already be in float format (e.g., from ADC)
 * @note MA-5 is applied internally
 */
void peak_detector_threshold_process(
    const float *ppg,
    uint16_t length,
    float fs,
    peak_threshold_result_t *result
);

/**
 * @brief Process a window of PPG data with custom parameters
 * 
 * @param ppg         Pointer to PPG samples (float)
 * @param length      Number of samples
 * @param fs          Sample rate in Hz
 * @param ma_window   Moving average window size (recommended: 3-7)
 * @param threshold_ratio  Ratio for adaptive threshold (recommended: 0.70-0.85)
 * @param refractory_ms    Refractory period in ms (recommended: 300-400)
 * @param result      Pointer to result structure (output)
 */
void peak_detector_threshold_process_custom(
    const float *ppg,
    uint16_t length,
    float fs,
    uint8_t ma_window,
    float threshold_ratio,
    float refractory_ms,
    peak_threshold_result_t *result
);

/**
 * @brief Get heart rate from RR intervals
 * 
 * @param result      Pointer to result structure
 * @return float      Mean heart rate in BPM, or 0 if no valid RR intervals
 */
float peak_detector_threshold_get_hr(const peak_threshold_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* PEAK_DETECTOR_THRESHOLD_H */