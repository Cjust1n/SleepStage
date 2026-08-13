/*
 * rr_history.h
 *
 * Rolling RR interval history (circular buffer).
 *
 * Maintains a sliding window of the most recent RR intervals
 * and their cumulative beat timestamps.
 *
 * Purpose:
 *   The training pipeline extracts HRV features from a 5-minute
 *   sliding window of heart-rate data. This buffer replicates
 *   that behaviour on the embedded device by keeping a continuous
 *   RR history across epoch boundaries.
 *
 * Buffer ownership:
 *   Statically allocated inside rr_history.c. No caller owns
 *   the data — access is via read-only pointers.
 *
 * Wrap-around behaviour:
 *   Power-of-2 capacity enables efficient bitmask wrap.
 *   When full, the oldest entry is silently overwritten.
 *
 * Overflow policy:
 *   The buffer never rejects a valid RR interval.
 *   Once capacity is reached, every insert evicts the oldest sample.
 *   This guarantees a continuous N-minute history at all times.
 */

#ifndef RR_HISTORY_H
#define RR_HISTORY_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Capacity: 512 RR intervals.
 *
 * Justification from sleep physiology:
 *   Sleep HR range: 40-100 BPM typical.
 *   Training window: 5 minutes (300 seconds).
 *   At 100 BPM: 500 intervals in 5 minutes.
 *   512 provides >= 5 minutes for all physiologically
 *   valid sleep heart rates.
 *
 *   Power of 2 enables bitmask wrap (faster than modulo).
 *
 * Memory: 512 * (4 + 2) = 3072 bytes + 4 bytes overhead ≈ 3.1 KB
 */
#define RR_HISTORY_CAPACITY   512
#define RR_HISTORY_MASK       (RR_HISTORY_CAPACITY - 1)

typedef struct
{
    uint32_t timestamp_ms;  /* cumulative milliseconds since session start */
    uint16_t rr_ms;         /* RR interval in milliseconds (0-65535) */
} rr_sample_t;

/*
 * Initialise the RR history buffer.
 * Call once at session start. Sets head and count to zero.
 * Does NOT clear the data array (unnecessary for correctness).
 */
void rr_history_init(void);

/*
 * Reset the RR history buffer.
 * Discards all stored intervals. head and count return to zero.
 * Use only for explicit session restart, never on epoch boundary.
 */
void rr_history_reset(void);

/*
 * Push one RR interval into the circular buffer.
 *
 * Parameters:
 *   rr_ms         — RR interval in milliseconds (0-65535)
 *   timestamp_ms  — cumulative beat timestamp in milliseconds
 *
 * Behaviour:
 *   If the buffer is not full, appends at head and increments count.
 *   If the buffer is full, overwrites the oldest entry and advances head.
 *   Never returns false — the buffer always accepts new data.
 */
void rr_history_add(uint16_t rr_ms,
                    uint32_t timestamp_ms);

/*
 * Return the number of valid RR intervals in the buffer.
 * Ranges from 0 (empty) to RR_HISTORY_CAPACITY (full).
 */
uint16_t rr_history_size(void);

/*
 * Returns true if the buffer has reached capacity.
 * When full, every subsequent add will evict the oldest sample.
 */
bool rr_history_is_full(void);

/*
 * Return the number of overwrites (samples evicted due to full buffer).
 * Monotonically increasing once the buffer reaches capacity.
 * Useful for validating that overflow behaviour is working correctly.
 */
uint32_t rr_history_get_overwrites(void);

/*
 * Return a contiguous view of the RR history.
 *
 * If the data is already contiguous in memory, returns a pointer
 * directly into the internal buffer (zero-copy path).
 *
 * If the data wraps around the circular boundary, copies into
 * the caller-provided scratch_buffer to linearise it, then
 * returns scratch_buffer.
 *
 * Parameters:
 *   scratch_buffer — caller-owned buffer of at least
 *                    RR_HISTORY_CAPACITY rr_sample_t elements.
 *                    Only used when linearisation is required.
 *
 * Returns:
 *   Pointer to contiguous rr_sample_t array of size rr_history_size(),
 *   or NULL if the buffer is empty.
 */
const rr_sample_t *rr_history_get_linear(
    rr_sample_t *scratch_buffer);

#ifdef __cplusplus
}
#endif

#endif /* RR_HISTORY_H */

