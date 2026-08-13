#ifndef ACCEL_BUFFER_H
#define ACCEL_BUFFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ACCEL_SAMPLE_RATE     50
#define ACCEL_EPOCH_SECONDS   30
#define ACCEL_BUFFER_SIZE     (ACCEL_SAMPLE_RATE * ACCEL_EPOCH_SECONDS)

typedef struct
{
    float x;
    float y;
    float z;
} accel_sample_t;

void accel_buffer_reset(void);

bool accel_buffer_add(float x,
                      float y,
                      float z);

bool accel_buffer_is_full(void);

uint16_t accel_buffer_size(void);

const accel_sample_t *accel_buffer_get(void);

#ifdef __cplusplus
}
#endif

#endif