#include "rr_buffer.h"

static float rr_buffer[RR_BUFFER_MAX];
static uint16_t rr_count = 0;

void rr_buffer_reset(void)
{
    rr_count = 0;
}

bool rr_buffer_add(float rr_ms)
{
    if(rr_count >= RR_BUFFER_MAX)
        return false;

    rr_buffer[rr_count++] = rr_ms;

    return true;
}

uint16_t rr_buffer_size(void)
{
    return rr_count;
}

const float* rr_buffer_get(void)
{
    return rr_buffer;
}