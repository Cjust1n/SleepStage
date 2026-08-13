/*
 * peak_detector_improved.c
 *
 * Implementation of the improved PPG peak detector.
 *
 * Key improvements over detect_peaks_inline() in cvapp_mb_cls.cpp:
 *   - Larger MA window (5 vs 3) → cleaner signal
 *   - Lower threshold (15 % vs 30 %) → detects smaller beats
 *   - Shorter refractory (250 ms vs 300 ms) → fewer missed peaks
 *   - Prominence filter → rejects motion spikes
 *   - Signal quality index → discard noisy epochs
 *
 * All memory is stack-allocated (VLA). The epoch window is typically
 * 750 samples (30 s @ 25 Hz), which is ~3 KB — comfortable on Cortex-M55.
 */

#include "peak_detector_improved.h"
#include <math.h>
#include <string.h>

/* =============================================================================
 * Static Helper Functions
 * ===========================================================================*/

/*
 * Apply a symmetric moving-average filter.
 *
 * For sample i, averages samples [i - window + 1 .. i] (causal).
 * This is simpler than a centred window and avoids look-ahead.
 */
static void apply_moving_average(const float *input,
                                 float *output,
                                 uint16_t length,
                                 uint8_t window)
{
    if (window > length)
        window = length;
    if (window < 1)
        window = 1;

    /* First sample — no averaging possible */
    output[0] = input[0];

    for (uint16_t i = 1; i < length; i++)
    {
        float sum   = 0.0f;
        uint8_t cnt = 0;
        uint16_t start = (i >= (uint16_t)window) ? (i - window + 1) : 0;

        for (uint16_t j = start; j <= i; j++)
        {
            sum += input[j];
            cnt++;
        }

        output[i] = sum / (float)cnt;
    }
}

/*
 * Check whether a candidate peak is "prominent" enough.
 *
 * Looks back `lookback` samples from the peak to find the lowest
 * trough. If the rise from trough to peak exceeds
 *   PROMINENCE_RATIO * threshold,
 * the peak is considered prominent — otherwise it is rejected
 * as noise / motion artifact.
 */
static bool is_peak_prominent(const float *ppg,
                              uint16_t idx,
                              float threshold,
                              uint16_t lookback)
{
    if (idx < lookback)
        lookback = idx;
    if (lookback < 1)
        return true;  /* too close to start — conservatively accept */

    float trough = ppg[idx];

    for (uint16_t i = idx - lookback; i < idx; i++)
    {
        if (ppg[i] < trough)
            trough = ppg[i];
    }

    float prominence    = ppg[idx] - trough;
    float min_prominent = threshold * PROMINENCE_RATIO_IMP;

    return (prominence > min_prominent);
}

/*
 * Core peak-finding logic.
 *
 * Scans through the filtered PPG signal, looking for local maxima
 * that exceed the adaptive threshold. For each candidate, applies:
 *   1. Local maximum check (greater than left and right neighbours)
 *   2. Above-threshold check
 *   3. Prominence check
 *   4. Refractory period check
 *
 * Returns the number of peaks found (written to peak_indices[]).
 */
static uint16_t find_peaks_improved(const float *ppg,
                                    uint16_t length,
                                    float threshold,
                                    uint16_t refractory_samples,
                                    uint16_t *peak_indices,
                                    uint16_t max_peaks)
{
    uint16_t count     = 0;
    uint16_t last_peak = 0;
    bool     in_peak   = false;
    uint16_t peak_start = 0;

    for (uint16_t i = 1; i < length - 1; i++)
    {
        bool is_local_max    = (ppg[i] > ppg[i - 1] && ppg[i] > ppg[i + 1]);
        bool is_above_thresh = (ppg[i] > threshold);

        if (is_local_max && is_above_thresh && !in_peak)
        {
            in_peak    = true;
            peak_start = i;
        }
        else if (in_peak && (ppg[i] < threshold || i == length - 2))
        {
            /* Find the true maximum within this peak block */
            uint16_t p    = peak_start;
            float    best = ppg[peak_start];

            for (uint16_t k = peak_start + 1; k < i; k++)
            {
                if (ppg[k] > best)
                {
                    best = ppg[k];
                    p    = k;
                }
            }

            /* Prominence check (look back 15 samples) */
            bool prominent = is_peak_prominent(ppg, p, threshold, 15);

            /* Refractory check */
            bool refractory_ok = (count == 0 ||
                                  (p - last_peak) >= refractory_samples);

            if (prominent && refractory_ok)
            {
                if (count < max_peaks)
                {
                    peak_indices[count++] = p;
                    last_peak             = p;
                }
            }

            in_peak = false;
        }
    }

    return count;
}

/*
 * Compute RR intervals from detected peak indices.
 *
 * Converts sample-index differences to milliseconds:
 *   RR_ms = (idx[n] - idx[n-1]) * 1000 / fs
 *
 * Applies physiological bounds [RR_MIN_MS_IMP, RR_MAX_MS_IMP].
 */
static uint16_t compute_rr_intervals(const uint16_t *peak_indices,
                                     uint16_t peak_count,
                                     float fs,
                                     float *rr_ms,
                                     uint16_t max_rr)
{
    uint16_t count = 0;

    for (uint16_t i = 1; i < peak_count && count < max_rr; i++)
    {
        float rr = (float)(peak_indices[i] - peak_indices[i - 1])
                 * 1000.0f / fs;

        if (rr >= (float)RR_MIN_MS_IMP && rr <= (float)RR_MAX_MS_IMP)
        {
            rr_ms[count++] = rr;
        }
    }

    return count;
}

/* =============================================================================
 * Public API
 * ===========================================================================*/

void peak_detector_improved_process(const float *ppg,
                                    uint16_t length,
                                    float fs,
                                    peak_improved_result_t *result)
{
    /* Clear output structure */
    memset(result, 0, sizeof(peak_improved_result_t));

    /* Input validation */
    if (length < 10 || fs <= 0.0f || ppg == NULL || result == NULL)
        return;

    /* ----------------------------------------------------------------
     * Step 1: Moving-average filter (window = 5)
     * ---------------------------------------------------------------- */
    float filtered[length];
    apply_moving_average(ppg, filtered, length, MA_WINDOW_IMP);

    /* ----------------------------------------------------------------
     * Step 2: Find min / max amplitude
     * ---------------------------------------------------------------- */
    float min_val = filtered[0];
    float max_val = filtered[0];

    for (uint16_t i = 1; i < length; i++)
    {
        if (filtered[i] < min_val) min_val = filtered[i];
        if (filtered[i] > max_val) max_val = filtered[i];
    }

    /* ----------------------------------------------------------------
     * Step 3: Adaptive threshold — 15 % of amplitude (was 30 %)
     * ---------------------------------------------------------------- */
    float amplitude = max_val - min_val;
    float threshold = min_val + THRESHOLD_RATIO_IMP * amplitude;

    /* ----------------------------------------------------------------
     * Step 4: Refractory period in samples
     * ---------------------------------------------------------------- */
    uint16_t refractory_samples = (uint16_t)(REFRACTORY_MS_IMP * fs / 1000.0f + 0.5f);
    if (refractory_samples < 1)
        refractory_samples = 1;

    /* ----------------------------------------------------------------
     * Step 5: Find peaks
     * ---------------------------------------------------------------- */
    uint16_t peak_indices_temp[PEAK_IMP_MAX];
    uint16_t peak_count = find_peaks_improved(filtered,
                                              length,
                                              threshold,
                                              refractory_samples,
                                              peak_indices_temp,
                                              PEAK_IMP_MAX);

    /* ----------------------------------------------------------------
     * Step 6: Copy results
     * ---------------------------------------------------------------- */
    for (uint16_t i = 0; i < peak_count && i < PEAK_IMP_MAX; i++)
        result->peak_indices[i] = peak_indices_temp[i];

    result->peak_count = peak_count;

    /* ----------------------------------------------------------------
     * Step 7: Compute RR intervals
     * ---------------------------------------------------------------- */
    result->rr_count = compute_rr_intervals(result->peak_indices,
                                            result->peak_count,
                                            fs,
                                            result->rr_ms,
                                            PEAK_IMP_MAX);

    /* ----------------------------------------------------------------
     * Step 8: Store metadata
     * ---------------------------------------------------------------- */
    result->threshold  = threshold;
    result->max_amp    = max_val;
    result->min_amp    = min_val;
    result->amplitude  = amplitude;

    /* ----------------------------------------------------------------
     * Step 9: Signal quality index
     * ---------------------------------------------------------------- */
    result->signal_quality = peak_detector_quality_index(
        filtered,
        length,
        result->peak_indices,
        result->peak_count);
}

float peak_detector_quality_index(const float *ppg,
                                  uint16_t length,
                                  const uint16_t *peaks,
                                  uint16_t peak_count)
{
    (void)ppg;
    (void)length;

    if (peak_count < 2)
        return 0.0f;

    /* Compute mean RR interval (in samples) */
    float sum_rr = 0.0f;
    for (uint16_t i = 1; i < peak_count; i++)
        sum_rr += (float)(peaks[i] - peaks[i - 1]);

    float mean_rr = sum_rr / (float)(peak_count - 1u);

    if (mean_rr < 1.0f)
        return 0.0f;

    /* Compute std of RR intervals (in samples) */
    float var_rr = 0.0f;
    for (uint16_t i = 1; i < peak_count; i++)
    {
        float diff = (float)(peaks[i] - peaks[i - 1]) - mean_rr;
        var_rr += diff * diff;
    }
    float std_rr = sqrtf(var_rr / (float)(peak_count - 1u));

    /* Quality = 1 - normalized std RR, clamped to [0, 1] */
    float quality = 1.0f - (std_rr / mean_rr);
    if (quality < 0.0f) quality = 0.0f;
    if (quality > 1.0f) quality = 1.0f;

    return quality;
}

float peak_detector_improved_get_hr(const peak_improved_result_t *result)
{
    if (result == NULL || result->rr_count == 0)
        return 0.0f;

    float sum_rr = 0.0f;
    for (uint16_t i = 0; i < result->rr_count; i++)
        sum_rr += result->rr_ms[i];

    float mean_rr = sum_rr / (float)result->rr_count;

    return (mean_rr > 0.0f) ? (60000.0f / mean_rr) : 0.0f;
}

