/*
 * cmsis_welch.c
 *
 * Welch power spectral density estimator.
 *
 * Uses a built-in radix-2 real FFT (decimation-in-time).
 *
 * The public API (cmsis_welch) is identical to the original
 * so hrv_features.c links without changes.
 *
 * FFT size: 256 (CMSIS_WELCH_FFT_SIZE).
 * Window:   Hamming.
 */

#include "cmsis_welch.h"

#include <math.h>
#include <string.h>
#include <stdint.h>

#define FFT_SIZE   CMSIS_WELCH_FFT_SIZE
#define HALF_SIZE  (FFT_SIZE / 2)

/* -----------------------------------------------------------------
 * Bit-reversal permutation table for FFT_SIZE = 256
 * ----------------------------------------------------------------- */
static const uint8_t bit_rev[FFT_SIZE] = {
      0,128, 64,192, 32,160, 96,224, 16,144, 80,208, 48,176,112,240,
      8,136, 72,200, 40,168,104,232, 24,152, 88,216, 56,184,120,248,
      4,132, 68,196, 36,164,100,228, 20,148, 84,212, 52,180,116,244,
     12,140, 76,204, 44,172,108,236, 28,156, 92,220, 60,188,124,252,
      2,130, 66,194, 34,162, 98,226, 18,146, 82,210, 50,178,114,242,
     10,138, 74,202, 42,170,106,234, 26,154, 90,218, 58,186,122,250,
      6,134, 70,198, 38,166,102,230, 22,150, 86,214, 54,182,118,246,
     14,142, 78,206, 46,174,110,238, 30,158, 94,222, 62,190,126,254,
      1,129, 65,193, 33,161, 97,225, 17,145, 81,209, 49,177,113,241,
      9,137, 73,201, 41,169,105,233, 25,153, 89,217, 57,185,121,249,
      5,133, 69,197, 37,165,101,229, 21,149, 85,213, 53,181,117,245,
     13,141, 77,205, 45,173,109,237, 29,157, 93,221, 61,189,125,253,
      3,131, 67,195, 35,163, 99,227, 19,147, 83,211, 51,179,115,243,
     11,139, 75,203, 43,171,107,235, 27,155, 91,219, 59,187,123,251,
      7,135, 71,199, 39,167,103,231, 23,151, 87,215, 55,183,119,247,
     15,143, 79,207, 47,175,111,239, 31,159, 95,223, 63,191,127,255
};

/* -----------------------------------------------------------------
 * Static buffers (BSS)
 * ----------------------------------------------------------------- */
static float window[FFT_SIZE];
static float fft_buffer[FFT_SIZE];
static float psd_acc[CMSIS_WELCH_BINS];
static uint8_t initialized = 0;

/* -----------------------------------------------------------------
 * Hamming window initialisation
 * ----------------------------------------------------------------- */
static void init_window(void)
{
    if (initialized)
        return;
    const float a0 = 0.54f;
    const float a1 = 0.46f;
    const float two_pi_over_n = 2.0f * 3.141592653589793f / (float)(FFT_SIZE - 1);
    for (uint32_t i = 0; i < FFT_SIZE; i++)
        window[i] = a0 - a1 * cosf(two_pi_over_n * (float)i);
    initialized = 1;
}

/* -----------------------------------------------------------------
 * In-place radix-2 complex FFT (decimation-in-time).
 * Interleaved complex array: real[0], imag[0], real[1], imag[1], ...
 * Length N must be power of 2.
 * ----------------------------------------------------------------- */
static void fft_complex(float *data, uint32_t n)
{
    for (uint32_t i = 0; i < n; i++)
    {
        uint32_t j = bit_rev[i];
        if (j > i)
        {
            float tr = data[2 * i];     float ti = data[2 * i + 1];
            data[2 * i]     = data[2 * j];
            data[2 * i + 1] = data[2 * j + 1];
            data[2 * j]     = tr;
            data[2 * j + 1] = ti;
        }
    }
    for (uint32_t len = 2; len <= n; len <<= 1)
    {
        uint32_t half = len >> 1;
        float theta = -3.141592653589793f / (float)half;
        float w_delta_re = cosf(theta);
        float w_delta_im = sinf(theta);
        float w_re = 1.0f, w_im = 0.0f;
        for (uint32_t j = 0; j < half; j++)
        {
            for (uint32_t i = j; i < n; i += len)
            {
                uint32_t ip = i + half;
                float t_re = w_re * data[2 * ip]     - w_im * data[2 * ip + 1];
                float t_im = w_re * data[2 * ip + 1] + w_im * data[2 * ip];
                data[2 * ip]     = data[2 * i]     - t_re;
                data[2 * ip + 1] = data[2 * i + 1] - t_im;
                data[2 * i]     += t_re;
                data[2 * i + 1] += t_im;
            }
            float tmp_re = w_re * w_delta_re - w_im * w_delta_im;
            w_im = w_re * w_delta_im + w_im * w_delta_re;
            w_re = tmp_re;
        }
    }
}

/* -----------------------------------------------------------------
 * Real FFT (packed format).
 * N must be power of 2.
 * Output: index 0 = DC, index N/2 = Nyquist, indices 1..N/2-1 = (real, imag)
 * ----------------------------------------------------------------- */
static void fft_real(float *data, uint32_t n)
{
    uint32_t m = n / 2;
    for (uint32_t i = 0; i < m; i++)
    {
        fft_buffer[2 * i]     = data[2 * i];
        fft_buffer[2 * i + 1] = data[2 * i + 1];
    }
    fft_complex(fft_buffer, m);
    float theta = -3.141592653589793f / (float)n;
    float w_delta_re = cosf(theta);
    float w_delta_im = sinf(theta);
    float w_re = 1.0f, w_im = 0.0f;
    data[0] = fft_buffer[0] + fft_buffer[1];
    data[1] = fft_buffer[0] - fft_buffer[1];
    for (uint32_t k = 1; k < m; k++)
    {
        float z_re = (fft_buffer[2 * k]     + fft_buffer[2 * (m - k)])     * 0.5f;
        float z_im = (fft_buffer[2 * k + 1] - fft_buffer[2 * (m - k) + 1]) * 0.5f;
        float z2_re = (fft_buffer[2 * (m - k) + 1] + fft_buffer[2 * k + 1]) * 0.5f;
        float z2_im = (fft_buffer[2 * (m - k)]     - fft_buffer[2 * k])     * 0.5f;
        data[2 * k]     = z_re + w_re * z2_re - w_im * z2_im;
        data[2 * k + 1] = z_im + w_re * z2_im + w_im * z2_re;
        float tmp = w_re * w_delta_re - w_im * w_delta_im;
        w_im = w_re * w_delta_im + w_im * w_delta_re;
        w_re = tmp;
    }
}

/* -----------------------------------------------------------------
 * Public API
 * ----------------------------------------------------------------- */
int cmsis_welch(
    const float *signal,
    uint32_t length,
    float fs,
    float *freq_out,
    float *psd_out)
{
    init_window();
    if (length < FFT_SIZE)
        return 0;
    memset(psd_acc, 0, sizeof(psd_acc));
    const uint32_t hop = HALF_SIZE;
    uint32_t segments = 0;
    for (uint32_t start = 0; start + FFT_SIZE <= length; start += hop)
    {
        for (uint32_t i = 0; i < FFT_SIZE; i++)
            fft_buffer[i] = signal[start + i] * window[i];
        fft_real(fft_buffer, FFT_SIZE);
        psd_acc[0] += fft_buffer[0] * fft_buffer[0];
        for (uint32_t k = 1; k < HALF_SIZE; k++)
        {
            float re = fft_buffer[2 * k];
            float im = fft_buffer[2 * k + 1];
            psd_acc[k] += re * re + im * im;
        }
        psd_acc[HALF_SIZE] += fft_buffer[1] * fft_buffer[1];
        segments++;
    }
    if (segments == 0)
        return 0;
    for (uint32_t k = 0; k <= HALF_SIZE; k++)
    {
        psd_out[k] = psd_acc[k] / (float)segments;
        freq_out[k] = (float)k * fs / (float)FFT_SIZE;
    }
    return CMSIS_WELCH_BINS;
}
