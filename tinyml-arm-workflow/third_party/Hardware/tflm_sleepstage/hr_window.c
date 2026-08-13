#include "hr_window.h"

static hr_sample_t hr_window[HR_WINDOW_SIZE];

static uint16_t count = 0;

void hr_window_reset(void)
{
    count = 0;
}

bool hr_window_add(float hr, float timestamp)
{
    if (count < HR_WINDOW_SIZE)
    {
        hr_window[count].hr = hr;
        hr_window[count].timestamp = timestamp;
        count++;
        return true;
    }

    // Sliding window
    for (uint16_t i = 1; i < HR_WINDOW_SIZE; i++)
    {
        hr_window[i - 1] = hr_window[i];
    }

    hr_window[HR_WINDOW_SIZE - 1].hr = hr;
    hr_window[HR_WINDOW_SIZE - 1].timestamp = timestamp;

    return true;
}

uint16_t hr_window_size(void)
{
    return count;
}

const hr_sample_t* hr_window_get(void)
{
    return hr_window;
}

bool hr_window_is_full(void)
{
    return (count >= HR_WINDOW_SIZE);
}