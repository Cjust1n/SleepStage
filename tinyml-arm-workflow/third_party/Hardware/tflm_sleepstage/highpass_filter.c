/*
 * highpass_filter.c
 *
 * First-order IIR high-pass filter implementation.
 *
 * Theory:
 *   This is a DC blocker using a single-pole IIR estimate of the
 *   DC component. It is equivalent to:
 *
 *     y[n] = x[n] - x[n-1] + R * y[n-1]   (where R = 1 - alpha)
 *
 *   but implemented in the DC-tracking form for numerical stability
 *   with integer/fixed-point inputs.
 *
 * Coefficient calculation:
 *   RC   = 1 / (2 * pi * cutoff)
 *   dt   = 1 / fs
 *   alpha = dt / (RC + dt)
 *
 * Example @ 25 Hz, cutoff = 0.5 Hz:
 *   RC    = 1 / (2 * pi * 0.5) = 0.3183 s
 *   dt    = 0.04 s
 *   alpha = 0.04 / (0.3183 + 0.04) = 0.1116
 */

#include "highpass_filter.h"
#include <string.h>

/* Pi constant (avoid math.h dependency) */
#define HPF_PI 3.14159265358979323846f

void highpass_filter_init(highpass_filter_t *filter,
                          float fs,
                          float cutoff)
{
    if (filter == NULL)
        return;

    if (fs <= 0.0f || cutoff <= 0.0f)
    {
        /* Invalid parameters: default to bypass (alpha = 1, no filtering) */
        filter->alpha = 1.0f;
        filter->dc_estimate = 0.0f;
        return;
    }

    /*
     * Guard against cutoff >= fs/2 (Nyquist).
     * If cutoff exceeds Nyquist, clamp to 0.45 * fs.
     */
    float nyquist = fs * 0.5f;
    float safe_cutoff = (cutoff < nyquist) ? cutoff : (nyquist * 0.9f);

    /*
     * Compute alpha for the DC-tracking filter.
     *
     *   RC   = 1 / (2 * pi * f_c)
     *   alpha = dt / (RC + dt)
     *         = 1 / (1 + RC / dt)
     *         = 1 / (1 + 1 / (2 * pi * f_c * dt))
     *         = 1 / (1 + fs / (2 * pi * f_c))
     *
     * This form avoids explicit RC calculation.
     */
    float ratio = fs / (2.0f * HPF_PI * safe_cutoff);
    filter->alpha = 1.0f / (1.0f + ratio);

    /* Initial DC estimate = 0 (filter will converge within ~1/alpha samples) */
    filter->dc_estimate = 0.0f;
}

float highpass_filter_process(highpass_filter_t *filter,
                              float input)
{
    if (filter == NULL)
        return input;

    /*
     * Update DC estimate (first-order low-pass on input):
     *   DC[n] = alpha * x[n] + (1 - alpha) * DC[n-1]
     */
    filter->dc_estimate = filter->alpha * input
                        + (1.0f - filter->alpha) * filter->dc_estimate;

    /*
     * High-pass output = input minus DC estimate:
     *   y[n] = x[n] - DC[n]
     */
    return input - filter->dc_estimate;
}

void highpass_filter_process_block(highpass_filter_t *filter,
                                   const float *input,
                                   float *output,
                                   uint16_t length)
{
    if (filter == NULL || input == NULL || output == NULL || length == 0)
        return;

    for (uint16_t i = 0; i < length; i++)
    {
        output[i] = highpass_filter_process(filter, input[i]);
    }
}

void highpass_filter_reset(highpass_filter_t *filter)
{
    if (filter == NULL)
        return;

    filter->dc_estimate = 0.0f;
    /* alpha remains unchanged (no need to re-initialise) */
}

