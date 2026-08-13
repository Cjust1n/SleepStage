#include "ppg_buffer.h"

static uint32_t ppg_buffer[PPG_EPOCH_SAMPLES];

static uint16_t write_index = 0;

void ppg_buffer_reset(void)
{
    write_index = 0;
}

bool ppg_buffer_add(uint32_t sample)
{
    if (write_index >= PPG_EPOCH_SAMPLES)
        return false;

    ppg_buffer[write_index++] = sample;

    return true;
}

bool ppg_buffer_is_full(void)
{
    return (write_index >= PPG_EPOCH_SAMPLES);
}

uint16_t ppg_buffer_size(void)
{
    return write_index;
}

const uint32_t* ppg_buffer_get(void)
{
    return ppg_buffer;
}