/*
 * highpass_filter.h
 *
 * First-order IIR high-pass filter.
 * Eliminates baseline drift / DC component from PPG signals
 * while preserving the AC cardiac component (0.5-5 Hz typical).
 *
 * Design:
 *   cutoff = 0.5 Hz @ 25 Hz sample rate
 *   Alpha = dt / (RC + dt)
 *   y[n] = x[n] - DC_estimate[n]
 *   DC_estimate[n] = alpha * x[n] + (1-alpha) * DC_estimate[n-1]
 *
 * Memory: ~16 bytes per instance (negligible).
 */

#ifndef HIGHPASS_FILTER_H
#define HIGHPASS_FILTER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * Data Structure
 * ===========================================================================*/

typedef struct {
    float alpha;           /* Filter coefficient (0..1) */
    float dc_estimate;     /* Running DC estimate */
} highpass_filter_t;

/* =============================================================================
 * Public API
 * ===========================================================================*/

/**
 * @brief Initialize high-pass filter.
 *
 * @param filter   Pointer to filter structure (must be non-NULL)
 * @param fs       Sample rate in Hz (e.g., 25.0)
 * @param cutoff   Cutoff frequency in Hz (recommended: 0.3 - 0.8)
 *
 * Example:
 *   highpass_filter_init(&my_filter, 25.0f, 0.5f); // 0.5 Hz @ 25 Hz
 */
void highpass_filter_init(highpass_filter_t *filter,
                          float fs,
                          float cutoff);

/**
 * @brief Process one sample through the filter.
 *
 * @param filter  Pointer to filter structure
 * @param input   Raw input sample
 * @return float  Filtered output (AC component, input minus DC estimate)
 */
float highpass_filter_process(highpass_filter_t *filter,
                              float input);

/**
 * @brief Process a block of samples (vectorised convenience).
 *
 * @param filter  Pointer to filter structure
 * @param input   Input samples (length >= 'length')
 * @param output  Output samples (may alias input)
 * @param length  Number of samples to process
 */
void highpass_filter_process_block(highpass_filter_t *filter,
                                   const float *input,
                                   float *output,
                                   uint16_t length);

/**
 * @brief Reset filter state to initial (zero DC estimate).
 */
void highpass_filter_reset(highpass_filter_t *filter);

#ifdef __cplusplus
}
#endif

#endif /* HIGHPASS_FILTER_H */

