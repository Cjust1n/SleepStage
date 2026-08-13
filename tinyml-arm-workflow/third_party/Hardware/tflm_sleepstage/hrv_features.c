/******************************************************************************
 * hrv_features.c
 *
 * Part A
 * Numerical helper functions
 ******************************************************************************/

#include "hrv_features.h"
#include "cmsis_welch.h"

#include <math.h>
#include <string.h>

#define RESAMPLE_FS        4.0f
#define MAX_INTERP_SAMPLES 1200

/******************************************************************************
 * Static Buffers
 ******************************************************************************/

static float diff_buffer[HRV_MAX_RR];
static float interp_buffer[HRV_MAX_INTERP];

/******************************************************************************
 * Mean
 ******************************************************************************/

static float compute_mean(
    const float *x,
    uint32_t n)
{
    if (x == NULL || n == 0)
        return 0.0f;

    float sum = 0.0f;

    for (uint32_t i = 0; i < n; i++)
    {
        sum += x[i];
    }

    return sum / (float)n;
}

/******************************************************************************
 * Standard Deviation
 *
 * Matches:
 * np.std(x)
 ******************************************************************************/

static float compute_std(
    const float *x,
    uint32_t n,
    float mean)
{
    if (x == NULL || n < 2)
        return 0.0f;

    float var = 0.0f;

    for (uint32_t i = 0; i < n; i++)
    {
        float d = x[i] - mean;
        var += d * d;
    }

    var /= (float)n;

    return sqrtf(var);
}

/******************************************************************************
 * RMSSD
 *
 * sqrt(mean(diff(RR)^2))
 ******************************************************************************/

static float compute_rmssd(
    const float *rr,
    uint32_t count)
{
    if (rr == NULL || count < 2)
        return 0.0f;

    float sum = 0.0f;

    uint32_t diff_count = count - 1;

    for (uint32_t i = 1; i < count; i++)
    {
        float d = rr[i] - rr[i - 1];

        diff_buffer[i - 1] = d;

        sum += d * d;
    }

    return sqrtf(sum / (float)diff_count);
}

/******************************************************************************
 * NN50
 *
 * Number of successive RR intervals
 * differing more than 50 ms.
 ******************************************************************************/

static float compute_nn50(
    const float *rr,
    uint32_t count)
{
    if (rr == NULL || count < 2)
        return 0.0f;

    uint32_t nn50 = 0;

    for (uint32_t i = 1; i < count; i++)
    {
        float d = fabsf(rr[i] - rr[i - 1]);

        if (d > 50.0f)
            nn50++;
    }

    return (float)nn50;
}

/******************************************************************************
 * SD2
 *
 * Python:
 *
 * sdnn = std(rr)
 * sdsd = std(diff)
 *
 * sd2 = sqrt(
 *          2*sdnn²
 *          -0.5*sdsd²
 *       )
 ******************************************************************************/

static float compute_sd2(
    const float *rr,
    uint32_t count)
{
    if (rr == NULL || count < 2)
        return 0.0f;

    float mean_rr =
        compute_mean(rr, count);

    float sdnn =
        compute_std(rr,
                    count,
                    mean_rr);

    uint32_t diff_count = count - 1;

    for (uint32_t i = 1; i < count; i++)
    {
        diff_buffer[i - 1] =
            rr[i] - rr[i - 1];
    }

    float mean_diff =
        compute_mean(diff_buffer,
                     diff_count);

    float sdsd =
        compute_std(diff_buffer,
                    diff_count,
                    mean_diff);

    float temp =
        2.0f * sdnn * sdnn
        - 0.5f * sdsd * sdsd;

    if (temp < 0.0f)
        temp = 0.0f;

    return sqrtf(temp);
}

/******************************************************************************
 * Linear Interpolation
 *
 * Equivalent to:
 * scipy.interpolate.interp1d(kind="linear")
 *
 * Input:
 *      rr_ms[]          : RR interval (ms)
 *      timestamp_ms[]   : Beat timestamps (ms)
 *
 * Output:
 *      interp_buffer[]  : RR resampled at 4 Hz
 *
 * Return:
 *      Number of interpolated samples
 ******************************************************************************/

static uint32_t interpolate_rr(
    const float *rr_ms,
    const uint32_t *timestamp_ms,
    uint32_t count)
{
    if (rr_ms == NULL ||
        timestamp_ms == NULL ||
        count < 2)
    {
        return 0;
    }

    const float dt = 1000.0f / RESAMPLE_FS;

    float t_start = (float)timestamp_ms[0];
    float t_end   = (float)timestamp_ms[count - 1];

    uint32_t out_idx = 0;
    uint32_t seg = 0;

    for (float t = t_start;
         t <= t_end && out_idx < HRV_MAX_INTERP;
         t += dt)
    {
        while ((seg < count - 2) &&
               ((float)timestamp_ms[seg + 1] < t))
        {
            seg++;
        }

        float t0 = (float)timestamp_ms[seg];
        float t1 = (float)timestamp_ms[seg + 1];

        float y0 = rr_ms[seg];
        float y1 = rr_ms[seg + 1];

        float value;

        if (fabsf(t1 - t0) < 1e-6f)
        {
            value = y0;
        }
        else
        {
            float alpha = (t - t0) / (t1 - t0);

            value =
                y0 +
                alpha * (y1 - y0);
        }

        interp_buffer[out_idx++] = value;
    }

    return out_idx;
}

/******************************************************************************
 * Remove Mean
 *
 * Equivalent to:
 *
 * ibi_interp - np.mean(ibi_interp)
 ******************************************************************************/

static void detrend(
    float *signal,
    uint32_t length)
{
    if (signal == NULL || length == 0)
        return;

    float mean =
        compute_mean(signal, length);

    for (uint32_t i = 0; i < length; i++)
    {
        signal[i] -= mean;
    }
}

/******************************************************************************
 * Prepare HRV Signal
 *
 * Pipeline:
 *
 * RR Buffer
 *      ↓
 * interpolate()
 *      ↓
 * detrend()
 *      ↓
 * interp_buffer[]
 *
 * Return:
 *      Number of valid interpolated samples
 ******************************************************************************/

static uint32_t prepare_hrv_signal(
    const float *rr_ms,
    const uint32_t *timestamp_ms,
    uint32_t count)
{
    uint32_t interp_count =
        interpolate_rr(
            rr_ms,
            timestamp_ms,
            count);

    if (interp_count < 8)
        return 0;

    detrend(
        interp_buffer,
        interp_count);

    return interp_count;
}

/******************************************************************************
 * Frequency Domain
 ******************************************************************************/

static void compute_frequency_domain(
    uint32_t interp_count,
    float *lf,
    float *hf,
    float *lf_hf)
{
    *lf = 0.0f;
    *hf = 0.0f;
    *lf_hf = 0.0f;

    if(interp_count < CMSIS_WELCH_FFT_SIZE)
        return;

    float freq[CMSIS_WELCH_BINS];
    float psd [CMSIS_WELCH_BINS];

    int bins =
        cmsis_welch(
            interp_buffer,
            interp_count,
            RESAMPLE_FS,
            freq,
            psd);

    if(bins <= 0)
        return;

    float lf_raw = 0.0f;
    float hf_raw = 0.0f;

    for(int i = 1; i < bins; i++)
    {
        float df = freq[i] - freq[i-1];

        if(freq[i] >= 0.04f &&
           freq[i] <  0.15f)
        {
            lf_raw += psd[i] * df;
        }

        if(freq[i] >= 0.15f &&
           freq[i] <  0.40f)
        {
            hf_raw += psd[i] * df;
        }
    }

    if(lf_raw < 0.0f)
        lf_raw = 0.0f;

    if(hf_raw < 0.0f)
        hf_raw = 0.0f;

    if(hf_raw < 1e-4f)
        return;

    *lf    = log1pf(lf_raw);
    *hf    = log1pf(hf_raw);
    *lf_hf = log1pf(lf_raw / hf_raw);
}

/******************************************************************************
 * Public API : hrv_features_extract
 *
 * Extracts HRV features from RR intervals and beat timestamps.
 * Features: mean_hr, rmssd, nn50, sd2, lf, hf, lf_hf
 *
 * Parameters:
 *   rr_ms[]           : RR intervals in milliseconds
 *   beat_timestamp_ms[]: Cumulative beat timestamps in milliseconds
 *   rr_count          : Number of RR intervals
 *   out               : Output features
 *
 * Returns:
 *   0 on success, -1 on failure
 ******************************************************************************/

int hrv_features_extract(
    const float *rr_ms,
    const uint32_t *beat_timestamp_ms,
    uint32_t rr_count,
    hrv_features_t *out)
{
    if (rr_ms == NULL || beat_timestamp_ms == NULL || out == NULL || rr_count < 2)
        return -1;

    // Clear output
    memset(out, 0, sizeof(hrv_features_t));

    // --- Time Domain ---

    // Mean HR: convert mean RR (ms) to BPM
    float mean_rr = compute_mean(rr_ms, rr_count);
    if (mean_rr > 0.0f)
        out->mean_hr = 60000.0f / mean_rr;

    out->rmssd = compute_rmssd(rr_ms, rr_count);
    out->nn50  = compute_nn50(rr_ms, rr_count);
    out->sd2   = compute_sd2(rr_ms, rr_count);

    // --- Frequency Domain ---

    uint32_t interp_count = prepare_hrv_signal(rr_ms, beat_timestamp_ms, rr_count);

    if (interp_count >= CMSIS_WELCH_FFT_SIZE)
    {
        compute_frequency_domain(interp_count, &out->lf, &out->hf, &out->lf_hf);
    }

    return 0;
}
