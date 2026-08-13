#ifndef TEMPORAL_FEATURES_H
#define TEMPORAL_FEATURES_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EPOCH_LENGTH_SEC 30

typedef struct
{
    float elapsed_sleep;
    float time_of_night;
    float relative_position;

} temporal_features_t;

/*
 * epoch          : current epoch index (0,1,2,...)
 * total_epochs   : total recording epochs
 * rec_start_unix : unix timestamp received from laptop
 */
void temporal_extract(
    uint32_t epoch,
    uint32_t total_epochs,
    uint32_t rec_start_unix,
    temporal_features_t *features);

#ifdef __cplusplus
}
#endif

#endif