/*
 * rr_history.c
 *
 * Rolling RR interval history — circular buffer implementation.
 *
 * See rr_history.h for design documentation.
 */

#include "rr_history.h"
#include "xprintf.h"

#include <string.h>

/******************************************************************************
 * Static Buffer
 *
 * Owned exclusively by this module.
 * No external code holds a pointer into this buffer
 * (the linearise function only returns the scratch buffer).
 ******************************************************************************/

static rr_sample_t g_buffer[RR_HISTORY_CAPACITY];

/*
 * Write index.
 * Always points to the next slot to write.
 * Modulo RR_HISTORY_MASK for O(1) wrap-around.
 */
static uint16_t g_head = 0;

/*
 * Number of valid entries in the buffer.
 * Ranges from 0 (empty) to RR_HISTORY_CAPACITY (full).
 */
static uint16_t g_count = 0;

/*
 * Monotonically increasing counter of overwritten entries.
 * Incremented each time an insert evicts the oldest sample.
 */
static uint32_t g_overwrites = 0;

/*
 * Debug counter — only print every 50 insertions
 * to keep the UART log readable.
 */
static uint16_t g_insertions_since_last_log = 0;

/*
 * Flag to ensure sizeof() is printed once at first init.
 */
static uint8_t g_sizeof_reported = 0;

/******************************************************************************
 * Public API
 ******************************************************************************/

void rr_history_init(void)
{
    g_head = 0;
    g_count = 0;
    g_overwrites = 0;
    g_insertions_since_last_log = 0;

    /*
     * Report sizeof(rr_sample_t) and total RAM once at startup.
     * This confirms the compiler's actual memory layout,
     * including any padding it may have inserted.
     */
    if (!g_sizeof_reported)
    {
        g_sizeof_reported = 1;

        uint32_t sample_size = (uint32_t)sizeof(rr_sample_t);
        uint32_t total_ram   = (uint32_t)sizeof(g_buffer);
        uint32_t capacity    = RR_HISTORY_CAPACITY;

        xprintf("RR_HISTORY: sizeof(rr_sample_t)=%u bytes"
                " (uint32_t ts=%u + uint16_t rr=%u + padding=%u)\r\n",
                (unsigned int)sample_size,
                (unsigned int)sizeof(uint32_t),
                (unsigned int)sizeof(uint16_t),
                (unsigned int)(sample_size - sizeof(uint32_t) - sizeof(uint16_t)));

        xprintf("RR_HISTORY: buffer[%u] = %u bytes total static RAM\r\n",
                (unsigned int)capacity,
                (unsigned int)total_ram);
    }
}

void rr_history_reset(void)
{
    g_head = 0;
    g_count = 0;
    g_overwrites = 0;
    g_insertions_since_last_log = 0;
}

void rr_history_add(uint16_t rr_ms,
                    uint32_t timestamp_ms)
{
    g_buffer[g_head].rr_ms = rr_ms;
    g_buffer[g_head].timestamp_ms = timestamp_ms;

    /*
     * Advance head with O(1) wrap-around.
     * RR_HISTORY_CAPACITY is a power of 2, so bitmask is equivalent
     * to modulo but without the division cost.
     */
    g_head = (g_head + 1) & RR_HISTORY_MASK;

    if (g_count < RR_HISTORY_CAPACITY)
    {
        g_count++;
    }
    else
    {
        /*
         * Buffer is full: increment overwrite counter.
         * The oldest entry (at the new head position) is about to be
         * overwritten on the next insert. Every insert from now on
         * evicts one old sample.
         */
        g_overwrites++;
    }

    /*
     * Enhanced debug log — every 50 insertions.
     * Prints:
     *   - history size (count of valid entries)
     *   - oldest and newest timestamps
     *   - total time span (duration) in seconds with one decimal
     *   - overwrite count
     *   - buffer full flag (0/1)
     */
    g_insertions_since_last_log++;

    if (g_insertions_since_last_log >= 50)
    {
        g_insertions_since_last_log = 0;

        uint16_t size   = g_count;
        uint32_t newest = g_count > 0 ? g_buffer[(g_head - 1) & RR_HISTORY_MASK].timestamp_ms : 0;
        uint32_t oldest;
        uint32_t overwrites = g_overwrites;
        uint8_t  full       = (g_count >= RR_HISTORY_CAPACITY) ? 1 : 0;

        /*
         * Compute oldest timestamp.
         *
         * If not full: oldest is at index 0.
         * If full: oldest is at g_head (the slot that will be overwritten next).
         */
        if (g_count == 0)
        {
            oldest = 0;
        }
        else if (g_count < RR_HISTORY_CAPACITY)
        {
            /* Contiguous from 0 */
            oldest = g_buffer[0].timestamp_ms;
        }
        else
        {
            /* Wrapped: oldest is at g_head */
            oldest = g_buffer[g_head].timestamp_ms;
        }

        uint32_t duration_ms = (newest > oldest) ? (newest - oldest) : 0;
        uint32_t duration_s  = duration_ms / 1000;
        uint32_t duration_d  = (duration_ms % 1000) / 100;  /* one decimal */

        xprintf("RR_HISTORY\r\n");
        xprintf("  size=%u\r\n",        (unsigned int)size);
        xprintf("  oldest=%u ms\r\n",   (unsigned int)oldest);
        xprintf("  newest=%u ms\r\n",   (unsigned int)newest);
        xprintf("  duration=%u.%u s\r\n", (unsigned int)duration_s, (unsigned int)duration_d);
        xprintf("  full=%u\r\n",        (unsigned int)full);
        xprintf("  overwrites=%u\r\n",  (unsigned int)overwrites);
    }
}

uint16_t rr_history_size(void)
{
    return g_count;
}

bool rr_history_is_full(void)
{
    return (g_count >= RR_HISTORY_CAPACITY);
}

uint32_t rr_history_get_overwrites(void)
{
    return g_overwrites;
}

const rr_sample_t *rr_history_get_linear(
    rr_sample_t *scratch_buffer)
{
    if (g_count == 0)
    {
        return NULL;
    }

    /*
     * Two cases:
     *
     * 1. Not full (g_count < CAPACITY)
     *    Data is stored contiguously starting at index 0.
     *    Return pointer directly into internal buffer (zero-copy).
     *
     * 2. Full (g_count == CAPACITY)
     *    Data wraps around. Oldest entry is at g_head,
     *    newest entry is at (g_head - 1) & MASK.
     *    Must linearise into scratch_buffer.
     */
    if (g_count < RR_HISTORY_CAPACITY)
    {
        /*
         * Contiguous path: data is at buffer[0 .. count-1].
         */
        return g_buffer;
    }

    /*
     * Wrapped path: copy two segments into scratch_buffer.
     *
     * Segment 1: g_buffer[g_head .. CAPACITY-1]
     *            (oldest entries, from head to end of array)
     *
     * Segment 2: g_buffer[0 .. g_head-1]
     *            (newest entries, from start of array to just before head)
     */
    uint16_t seg1_len = RR_HISTORY_CAPACITY - g_head;
    uint16_t seg2_len = g_head;

    if (seg1_len > 0)
    {
        (void)memcpy(scratch_buffer,
                     &g_buffer[g_head],
                     seg1_len * sizeof(rr_sample_t));
    }

    if (seg2_len > 0)
    {
        (void)memcpy(&scratch_buffer[seg1_len],
                     g_buffer,
                     seg2_len * sizeof(rr_sample_t));
    }

    return scratch_buffer;
}
