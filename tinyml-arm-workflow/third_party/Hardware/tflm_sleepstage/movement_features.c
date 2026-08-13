#include "movement_features.h"

#include <math.h>
#include <string.h>

void movement_extract(
    const accel_sample_t *samples,
    uint16_t length,
    movement_features_t *features)
{
    if (samples == NULL || features == NULL || length == 0)
        return;

    memset(features, 0, sizeof(movement_features_t));

    float sum = 0.0f;
    float energy = 0.0f;

    //--------------------------------------------------
    // Pass 1
    // magnitude
    //--------------------------------------------------

    static float mag_buffer[ACCEL_BUFFER_SIZE];

    for (uint16_t i = 0; i < length; i++)
    {
        float mag =
            sqrtf(
                samples[i].x * samples[i].x +
                samples[i].y * samples[i].y +
                samples[i].z * samples[i].z);

        mag_buffer[i] = mag;

        sum += mag;
        energy += mag * mag;
    }

    float mean = sum / (float)length;

    //--------------------------------------------------
    // Pass 2
    //--------------------------------------------------

    float variance_sum = 0.0f;
    uint32_t movement_count = 0;
    uint32_t zero_crossing = 0;

    float prev_mag = mag_buffer[0];
    float prev_centered = prev_mag - mean;
    float jerk_sum = 0.0f;

    for (uint16_t i = 0; i < length; i++)
    {
        float mag = mag_buffer[i];

        float diff = mag - mean;

        variance_sum += diff * diff;

        if (i > 0)
        {
            if (fabsf(mag - prev_mag) > 0.05f)
                movement_count++;

            // |diff(mag)| for acceleration_jerk = mean(|diff(mag)|)
            jerk_sum += fabsf(mag - prev_mag);

            float centered = mag - mean;

            if (prev_centered * centered < 0.0f)
                zero_crossing++;

            prev_centered = centered;
        }

        prev_mag = mag;
    }

    features->mean_acc = mean;
    features->std_acc = sqrtf(variance_sum / (float)length);
    features->energy = energy;

    // rms = sqrt(mean(mag^2)) = sqrt(energy / N)
    features->rms = (length > 0)
        ? sqrtf(energy / (float)length)
        : 0.0f;

    // acceleration_jerk = mean(|diff(mag)|) over the epoch
    features->acceleration_jerk = (length > 1)
        ? jerk_sum / (float)(length - 1)
        : 0.0f;
    features->movement_count = (float)movement_count;
    features->zero_crossing = (float)zero_crossing;
}