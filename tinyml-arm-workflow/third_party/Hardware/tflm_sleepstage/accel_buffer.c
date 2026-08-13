#include "accel_buffer.h"

static accel_sample_t g_buffer[ACCEL_BUFFER_SIZE];

static uint16_t g_index = 0;

void accel_buffer_reset(void)
{
    g_index = 0;
}

bool accel_buffer_add(float x,
                      float y,
                      float z)
{
    if (g_index >= ACCEL_BUFFER_SIZE)
    {
        return false;
    }

    g_buffer[g_index].x = x;
    g_buffer[g_index].y = y;
    g_buffer[g_index].z = z;

    g_index++;

    return true;
}

bool accel_buffer_is_full(void)
{
    return (g_index >= ACCEL_BUFFER_SIZE);
}

uint16_t accel_buffer_size(void)
{
    return g_index;
}

const accel_sample_t *accel_buffer_get(void)
{
    return g_buffer;
}