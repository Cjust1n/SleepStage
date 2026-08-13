// peak_detector.h

#ifndef PEAK_DETECTOR_H
#define PEAK_DETECTOR_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * Configuration Parameters
 * ===========================================================================*/

/* Maximum number of peaks to detect per epoch */
#define PEAK_MAX                    16

/* Moving average window size (samples) */
#define MA_WINDOW                   3    /* Dikurangi dari 5 ke 3 (120ms @25Hz) */

/* Threshold ratio for peak detection (0.0 - 1.0) */
#define THRESHOLD_RATIO             0.3  /* Dikurangi dari 0.5 ke 0.3 */

/* Refractory period (ms) - minimum time between peaks */
#define REFRACTORY_MS               300  /* Dikurangi dari 200 ke 300ms */

/* Physiological RR interval limits (ms) */
#define RR_MIN_MS                   300  /* 200 bpm max */
#define RR_MAX_MS                   1200 /* 50 bpm min */

/* =============================================================================
 * Data Structures
 * ===========================================================================*/

typedef struct {
    uint16_t peak_indices[PEAK_MAX];
    float rr_ms[PEAK_MAX];
    uint16_t peak_count;
    uint16_t rr_count;
    float threshold;
    float max_amp;
    float min_amp;
} peak_threshold_result_t;

/* =============================================================================
 * Public API
 * ===========================================================================*/

void peak_detector_threshold_process(
    const float *ppg,
    uint16_t length,
    float fs,
    peak_threshold_result_t *result
);

void peak_detector_threshold_process_custom(
    const float *ppg,
    uint16_t length,
    float fs,
    uint8_t ma_window,
    float threshold_ratio,
    float refractory_ms,
    peak_threshold_result_t *result
);

float peak_detector_threshold_get_hr(const peak_threshold_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* PEAK_DETECTOR_H */c