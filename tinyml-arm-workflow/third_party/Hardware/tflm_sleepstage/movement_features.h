#ifndef MOVEMENT_FEATURES_H
#define MOVEMENT_FEATURES_H

#include <stdint.h>
#include "accel_buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float mean_acc;
    float std_acc;
    float energy;
    float rms;
    float acceleration_jerk;
    float movement_count;
    float zero_crossing;
} movement_features_t;

/*
 * samples : pointer dari accel_buffer_get()
 * length  : jumlah sample (1500)
 * features: output feature
 */
void movement_extract(
    const accel_sample_t *samples,
    uint16_t length,
    movement_features_t *features);

#ifdef __cplusplus
}
#endif

#endif