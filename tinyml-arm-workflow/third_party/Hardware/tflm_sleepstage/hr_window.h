#ifndef HR_WINDOW_H
#define HR_WINDOW_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// 5 minutes / 30 seconds = 10 epochs
#define HR_WINDOW_SIZE 10

typedef struct
{
    float hr;            // BPM
    float timestamp;     // seconds since recording start
} hr_sample_t;

void hr_window_reset(void);

bool hr_window_add(float hr, float timestamp);

uint16_t hr_window_size(void);

const hr_sample_t* hr_window_get(void);

bool hr_window_is_full(void);

#ifdef __cplusplus
}
#endif

#endif