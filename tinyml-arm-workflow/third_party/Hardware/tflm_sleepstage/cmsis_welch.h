#ifndef CMSIS_WELCH_H
#define CMSIS_WELCH_H

#define CMSIS_WELCH_FFT_SIZE 256
#define CMSIS_WELCH_BINS (CMSIS_WELCH_FFT_SIZE/2+1)

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Compute Welch PSD.
 *
 * signal      : detrended signal
 * length      : number of samples
 * fs          : sampling frequency
 *
 * freq_out    : frequency axis
 * psd_out     : PSD values
 *
 * return:
 *      number of frequency bins
 */
int cmsis_welch(
    const float *signal,
    uint32_t length,
    float fs,
    float *freq_out,
    float *psd_out);

#ifdef __cplusplus
}
#endif

#endif