#include "temporal_features.h"

#include <stddef.h>

/*
 * Convert Unix timestamp
 * --> hour + minute/60
 *
 * Uses Unix epoch.
 * Timezone offset is configurable.
 *
 * WIB (Indonesia) = UTC+7
 * Training pipeline (SleepStage/FeatureExtraction/temporal.py)
 * uses America/New_York (UTC-4). Ubah di sini agar sesuai lokasi.
 */

#define LOCAL_TIMEZONE_OFFSET_HOURS (+7)

/**********************************************************************
 * Convert Unix timestamp to hour of day.
 **********************************************************************/
static float unix_to_hour(uint32_t unix_time)
{
    int32_t local =
        (int32_t)unix_time +
        LOCAL_TIMEZONE_OFFSET_HOURS * 3600;

    if(local < 0)
        local = 0;

    uint32_t seconds_today =
        (uint32_t)local % 86400;

    uint32_t hour =
        seconds_today / 3600;

    uint32_t minute =
        (seconds_today % 3600) / 60;

    return (float)hour +
           (float)minute / 60.0f;
}

/**********************************************************************
 * Feature extraction
 **********************************************************************/
void temporal_extract(
    uint32_t epoch,
    uint32_t total_epochs,
    uint32_t rec_start_unix,
    temporal_features_t *features)
{
    if(features == NULL)
        return;

    //-----------------------------
    // elapsed_sleep (minutes)
    //-----------------------------

    features->elapsed_sleep =
        (float)(epoch * EPOCH_LENGTH_SEC) / 60.0f;

    //-----------------------------
    // relative_position
    //-----------------------------

    if(total_epochs > 1)
    {
        features->relative_position =
            (float)epoch /
            (float)(total_epochs - 1);
    }
    else
    {
        features->relative_position = 0.0f;
    }

    //-----------------------------
    // time_of_night
    //-----------------------------

    uint32_t current_time =
        rec_start_unix +
        epoch * EPOCH_LENGTH_SEC;

    features->time_of_night =
        unix_to_hour(current_time);
}