#ifndef PPG_BUFFER_H
#define PPG_BUFFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PPG_EPOCH_SAMPLES    750   /* 30 sec x 25 Hz (actual MAX30102 effective rate) */

void ppg_buffer_reset(void);

bool ppg_buffer_add(uint32_t sample);

bool ppg_buffer_is_full(void);

uint16_t ppg_buffer_size(void);

const uint32_t* ppg_buffer_get(void);

#ifdef __cplusplus
}
#endif

#endif