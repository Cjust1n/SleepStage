#ifndef RR_BUFFER_H
#define RR_BUFFER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RR_BUFFER_MAX 128

void rr_buffer_reset(void);

bool rr_buffer_add(float rr_ms);

uint16_t rr_buffer_size(void);

const float* rr_buffer_get(void);

#ifdef __cplusplus
}
#endif

#endif