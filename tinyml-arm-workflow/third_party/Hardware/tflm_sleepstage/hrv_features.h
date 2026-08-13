#ifndef HRV_FEATURES_H
#define HRV_FEATURES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HRV_MAX_RR          1024
#define HRV_MAX_INTERP      1200

typedef struct
{
    float mean_hr;

    float rmssd;

    float nn50;

    float sd2;

    float lf;

    float hf;

    float lf_hf;

} hrv_features_t;

int hrv_features_extract(
    const float *rr_ms,
    const uint32_t *beat_timestamp_ms,
    uint32_t rr_count,
    hrv_features_t *out);

#ifdef __cplusplus
}
#endif

#endif
