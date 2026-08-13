/*
 * test_rr_history.c
 *
 * Comprehensive unit test suite for rr_history circular buffer.
 *
 * Compiles and runs entirely on a Linux desktop using GCC/Clang.
 * Zero dependencies on:
 *   - Cortex-M / Arm CMSIS
 *   - WE2 SDK / HX-Driver
 *   - UART / MAX30102 / MPU6050
 *   - TensorFlow Lite / Ethos-U55
 *
 * Compilation:
 *   gcc -Wall -Wextra -Wpedantic -Werror -std=c99       \
 *       -I. -I..                                         \
 *       test_rr_history.c ../rr_history.c                \
 *       -o test_rr_history
 *
 * Expected output:
 *   TEST 1  (empty buffer):         PASS  (or FAIL)
 *   ...
 *   TEST 13 (stress test):          PASS
 *   ========================================
 *   Result: 13/13 passed
 *
 * Author: Test suite generator
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

/* Include the header under test */
#include "../rr_history.h"

/*
 * ------------------------------------------------------------------
 * Internal state accessors (needed because g_buffer, g_head, g_count,
 * g_overwrites are static in rr_history.c).
 *
 * We cannot inspect them directly. Instead we verify behaviour
 * through the public API: rr_history_size(), rr_history_is_full(),
 * rr_history_get_overwrites(), rr_history_get_linear().
 * ------------------------------------------------------------------
 */

/*
 * Test counters
 */
static int g_tests_passed = 0;
static int g_tests_failed = 0;

static void test_begin(const char *name)
{
    printf("  TEST: %-45s ", name);
    fflush(stdout);
}

static void test_pass(void)
{
    g_tests_passed++;
    printf("PASS\n");
}

static void test_fail(const char *reason)
{
    g_tests_failed++;
    printf("FAIL  [%s]\n", reason);
    fflush(stdout);
}

/*
 * Helper: check two uint16_t values.
 */
static int check_u16(uint16_t got, uint16_t expected, const char *label)
{
    if (got != expected)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: got %u, expected %u",
                 label, (unsigned)got, (unsigned)expected);
        test_fail(buf);
        return 0;
    }
    return 1;
}

/*
 * Helper: check two uint32_t values.
 */
static int check_u32(uint32_t got, uint32_t expected, const char *label)
{
    if (got != expected)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: got %u, expected %u",
                 label, (unsigned)got, (unsigned)expected);
        test_fail(buf);
        return 0;
    }
    return 1;
}

/*
 * Helper: check a boolean.
 */
static int check_bool(bool got, bool expected, const char *label)
{
    if (got != expected)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: got %d, expected %d",
                 label, (int)got, (int)expected);
        test_fail(buf);
        return 0;
    }
    return 1;
}

/*
 * Helper: check pointer non-null.
 */
static int check_not_null(const void *ptr, const char *label)
{
    if (ptr == NULL)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: unexpected NULL", label);
        test_fail(buf);
        return 0;
    }
    return 1;
}

/*
 * Helper: check pointer is NULL.
 */
static int check_null(const void *ptr, const char *label)
{
    if (ptr != NULL)
    {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s: expected NULL, got non-NULL", label);
        test_fail(buf);
        return 0;
    }
    return 1;
}

/*
 * ==================================================================
 * TEST 1 — Empty buffer
 *
 * After init, the buffer must be:
 *   - size == 0
 *   - not full
 *   - overwrites == 0
 *   - get_linear() returns NULL
 * ==================================================================
 */
static void test_empty_buffer(void)
{
    test_begin("Empty buffer");

    rr_history_init();

    if (!check_u16(rr_history_size(), 0,         "size"))      return;
    if (!check_bool(rr_history_is_full(), false,  "is_full"))   return;
    if (!check_u32(rr_history_get_overwrites(), 0, "overwrites")) return;

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *result = rr_history_get_linear(scratch);
    if (!check_null(result, "get_linear")) return;

    test_pass();
}

/*
 * ==================================================================
 * TEST 2 — Single insertion
 *
 * After push:
 *   - size == 1
 *   - not full
 *   - overwrites == 0
 *   - get_linear() returns non-NULL
 *   - timestamp preserved
 *   - rr value preserved
 * ==================================================================
 */
static void test_single_insertion(void)
{
    test_begin("Single insertion");

    rr_history_init();

    rr_history_add(800, 1000);

    if (!check_u16(rr_history_size(), 1,        "size"))        return;
    if (!check_bool(rr_history_is_full(), false, "is_full"))    return;
    if (!check_u32(rr_history_get_overwrites(), 0, "overwrites")) return;

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    if (!check_u16(data[0].rr_ms,         800,  "rr_ms[0]"))   return;
    if (!check_u32(data[0].timestamp_ms,  1000, "ts[0]"))      return;

    test_pass();
}

/*
 * ==================================================================
 * TEST 3 — Multiple insertions (before buffer is full)
 *
 * Insert N < CAPACITY entries. Verify:
 *   - size == N
 *   - not full
 *   - insertion order preserved
 *   - timestamps monotonic
 *   - rr values preserved
 * ==================================================================
 */
static void test_multiple_insertions(void)
{
    test_begin("Multiple insertions (10 entries)");

    rr_history_init();

    static const uint16_t rr_vals[]  = {800, 820, 790, 810, 805, 830, 780, 815, 795, 825};
    static const uint32_t ts_vals[]  = {1000, 1820, 2610, 3420, 4225, 5055, 5835, 6650, 7445, 8270};
    const int N = 10;

    for (int i = 0; i < N; i++)
    {
        rr_history_add(rr_vals[i], ts_vals[i]);
    }

    if (!check_u16(rr_history_size(), N,          "size"))       return;
    if (!check_bool(rr_history_is_full(), false,   "is_full"))   return;
    if (!check_u32(rr_history_get_overwrites(), 0, "overwrites")) return;

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    for (int i = 0; i < N; i++)
    {
        char label[64];
        snprintf(label, sizeof(label), "rr_ms[%d]", i);
        if (!check_u16(data[i].rr_ms, rr_vals[i], label)) return;

        snprintf(label, sizeof(label), "ts[%d]", i);
        if (!check_u32(data[i].timestamp_ms, ts_vals[i], label)) return;
    }

    test_pass();
}

/*
 * ==================================================================
 * TEST 4 — Buffer reaches capacity
 *
 * Insert exactly CAPACITY entries. Verify:
 *   - size == CAPACITY
 *   - is_full == true
 *   - overwrites == 0  (no overwrites yet — exactly filled)
 *   - all timestamps monotonic
 * ==================================================================
 */
static void test_buffer_reaches_capacity(void)
{
    test_begin("Buffer reaches capacity");

    rr_history_init();

    for (uint16_t i = 0; i < RR_HISTORY_CAPACITY; i++)
    {
        rr_history_add((uint16_t)(800 + (i % 50)), (uint32_t)(i * 1000));
    }

    if (!check_u16(rr_history_size(), RR_HISTORY_CAPACITY, "size"))     return;
    if (!check_bool(rr_history_is_full(), true,          "is_full"))    return;
    if (!check_u32(rr_history_get_overwrites(), 0,       "overwrites")) return;

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    /* Verify monotonic timestamps and correct count */
    uint32_t prev_ts = data[0].timestamp_ms;
    for (uint16_t i = 1; i < RR_HISTORY_CAPACITY; i++)
    {
        if (data[i].timestamp_ms <= prev_ts)
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "timestamp not monotonic at index %u: %u <= %u",
                     (unsigned)i, (unsigned)data[i].timestamp_ms, (unsigned)prev_ts);
            test_fail(buf);
            return;
        }
        prev_ts = data[i].timestamp_ms;
    }

    test_pass();
}

/*
 * ==================================================================
 * TEST 5 — Buffer overflow (one extra insertion)
 *
 * Fill buffer, then add one more. Verify:
 *   - size == CAPACITY
 *   - is_full == true
 *   - overwrites == 1 (one eviction)
 *   - oldest entry discarded
 *   - newest entry is the overflow insertion
 * ==================================================================
 */
static void test_buffer_overflow(void)
{
    test_begin("Buffer overflow (1 extra)");

    rr_history_init();

    /* Fill buffer with entries 0..CAPACITY-1 */
    for (uint16_t i = 0; i < RR_HISTORY_CAPACITY; i++)
    {
        rr_history_add((uint16_t)(800 + i), (uint32_t)(i * 1000));
    }

    /* Add one more: entry CAPACITY */
    rr_history_add(999, (uint32_t)(RR_HISTORY_CAPACITY * 1000));

    if (!check_u16(rr_history_size(), RR_HISTORY_CAPACITY, "size"))     return;
    if (!check_bool(rr_history_is_full(), true,          "is_full"))    return;
    if (!check_u32(rr_history_get_overwrites(), 1,       "overwrites")) return;

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    /* Oldest entry should be index 1 (was 0, but 0 got evicted) */
    if (!check_u16(data[0].rr_ms, (uint16_t)(800 + 1), "oldest rr")) return;
    if (!check_u32(data[0].timestamp_ms, 1000, "oldest ts")) return;

    /* Newest entry should be 999 at CAPACITY*1000 */
    if (!check_u16(data[RR_HISTORY_CAPACITY - 1].rr_ms, 999, "newest rr")) return;
    if (!check_u32(data[RR_HISTORY_CAPACITY - 1].timestamp_ms,
                   (uint32_t)(RR_HISTORY_CAPACITY * 1000), "newest ts")) return;

    test_pass();
}

/*
 * ==================================================================
 * TEST 6 — Wrap-around (complete overwrite cycle)
 *
 * Insert 2*CAPACITY entries. The buffer should have wrapped
 * around completely. Verify:
 *   - size == CAPACITY
 *   - overwrites == CAPACITY
 *   - the oldest surviving entry is entry CAPACITY
 *     (the first CAPACITY entries were evicted)
 * ==================================================================
 */
static void test_wrap_around(void)
{
    test_begin("Wrap-around (2x capacity)");

    rr_history_init();

    const uint32_t total = 2 * RR_HISTORY_CAPACITY;

    for (uint32_t i = 0; i < total; i++)
    {
        rr_history_add((uint16_t)(i & 0xFFFF), (uint32_t)(i * 1000));
    }

    if (!check_u16(rr_history_size(), RR_HISTORY_CAPACITY, "size"))     return;
    if (!check_bool(rr_history_is_full(), true,          "is_full"))    return;
    if (!check_u32(rr_history_get_overwrites(),
                   (uint32_t)RR_HISTORY_CAPACITY, "overwrites")) return;

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    /* Oldest surviving entry should be the one at index CAPACITY */
    if (!check_u16(data[0].rr_ms, (uint16_t)RR_HISTORY_CAPACITY, "oldest rr after wrap")) return;
    if (!check_u32(data[0].timestamp_ms,
                   (uint32_t)(RR_HISTORY_CAPACITY * 1000), "oldest ts after wrap")) return;

    /* Newest entry should be the last inserted */
    if (!check_u16(data[RR_HISTORY_CAPACITY - 1].rr_ms,
                   (uint16_t)(total - 1), "newest rr after wrap")) return;
    if (!check_u32(data[RR_HISTORY_CAPACITY - 1].timestamp_ms,
                   (uint32_t)((total - 1) * 1000), "newest ts after wrap")) return;

    /* Verify ALL timestamps are monotonic */
    uint32_t prev_ts = data[0].timestamp_ms;
    for (uint16_t i = 1; i < RR_HISTORY_CAPACITY; i++)
    {
        if (data[i].timestamp_ms <= prev_ts)
        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "timestamp not monotonic after wrap at index %u: %u <= %u",
                     (unsigned)i, (unsigned)data[i].timestamp_ms, (unsigned)prev_ts);
            test_fail(buf);
            return;
        }
        prev_ts = data[i].timestamp_ms;
    }

    test_pass();
}

/*
 * ==================================================================
 * TEST 7 — Linearization order
 *
 * Verify that get_linear() returns data in insertion order:
 *   - Pre-fill buffer to capacity
 *   - Add entries that cause wrap
 *   - Linearize and check order matches insertion sequence
 * ==================================================================
 */
static void test_linearization_order(void)
{
    test_begin("Linearization order");

    rr_history_init();

    /* Fill to capacity */
    for (uint16_t i = 0; i < RR_HISTORY_CAPACITY; i++)
    {
        rr_history_add((uint16_t)(100 + i), (uint32_t)(i * 100));
    }

    /* Add a few more to cause partial wrap (head != 0) */
    for (uint16_t i = 0; i < 10; i++)
    {
        uint16_t idx = RR_HISTORY_CAPACITY + i;
        rr_history_add((uint16_t)(100 + idx), (uint32_t)(idx * 100));
    }

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    /* The linearised data should be in insertion order.
     * Oldest = index 10 (since indices 0..9 were evicted).
     * Newest = index CAPACITY+9 */
    for (uint16_t i = 0; i < RR_HISTORY_CAPACITY; i++)
    {
        uint16_t expected_idx = 10 + i;   /* first 10 evicted */
        uint16_t expected_rr  = (uint16_t)(100 + expected_idx);
        uint32_t expected_ts  = (uint32_t)(expected_idx * 100);

        char label[64];
        snprintf(label, sizeof(label), "order[%u] rr", (unsigned)i);
        if (!check_u16(data[i].rr_ms, expected_rr, label)) return;

        snprintf(label, sizeof(label), "order[%u] ts", (unsigned)i);
        if (!check_u32(data[i].timestamp_ms, expected_ts, label)) return;
    }

    test_pass();
}

/*
 * ==================================================================
 * TEST 8 — Timestamp preservation
 *
 * Verify that timestamp values are bit-exact preserved
 * through the circular buffer, including after wrap.
 * ==================================================================
 */
static void test_timestamp_preservation(void)
{
    test_begin("Timestamp preservation");

    rr_history_init();

    /* Insert with known, non-trivial timestamps */
    for (uint32_t i = 0; i < RR_HISTORY_CAPACITY + 20; i++)
    {
        /* Use a distinctive pattern: timestamp = i * 1000 + 12345 */
        rr_history_add((uint16_t)(500 + (i % 200)), i * 1000 + 12345);
    }

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    for (uint16_t i = 0; i < RR_HISTORY_CAPACITY; i++)
    {
        uint32_t expected_ts = (uint32_t)((20 + i) * 1000 + 12345);
        char label[64];
        snprintf(label, sizeof(label), "ts preserve[%u]", (unsigned)i);
        if (!check_u32(data[i].timestamp_ms, expected_ts, label)) return;
    }

    test_pass();
}

/*
 * ==================================================================
 * TEST 9 — RR preservation
 *
 * Verify that RR interval values are bit-exact preserved
 * through the circular buffer, including after wrap.
 * ==================================================================
 */
static void test_rr_preservation(void)
{
    test_begin("RR preservation");

    rr_history_init();

    /* Insert with a known non-trivial pattern */
    for (uint32_t i = 0; i < RR_HISTORY_CAPACITY + 30; i++)
    {
        uint16_t rr_val = (uint16_t)((i * 7 + 13) % 1000 + 300); /* 300..1299 */
        rr_history_add(rr_val, i * 1000);
    }

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear")) return;

    for (uint16_t i = 0; i < RR_HISTORY_CAPACITY; i++)
    {
        uint32_t orig_idx = 30 + i;
        uint16_t expected_rr = (uint16_t)((orig_idx * 7 + 13) % 1000 + 300);

        char label[64];
        snprintf(label, sizeof(label), "rr preserve[%u]", (unsigned)i);
        if (!check_u16(data[i].rr_ms, expected_rr, label)) return;
    }

    test_pass();
}

/*
 * ==================================================================
 * TEST 10 — Reset
 *
 * After reset:
 *   - size == 0
 *   - not full
 *   - overwrites == 0
 *   - get_linear() returns NULL
 *   - re-filling works correctly
 * ==================================================================
 */
static void test_reset(void)
{
    test_begin("Reset");

    rr_history_init();

    /* Fill and overflow */
    for (uint32_t i = 0; i < RR_HISTORY_CAPACITY + 5; i++)
    {
        rr_history_add((uint16_t)(800 + (i % 50)), i * 1000);
    }

    /* Verify it's in a non-trivial state */
    if (rr_history_size() != RR_HISTORY_CAPACITY)
    {
        test_fail("buffer should be full before reset");
        return;
    }

    /* Reset */
    rr_history_reset();

    if (!check_u16(rr_history_size(), 0,         "size after reset"))           return;
    if (!check_bool(rr_history_is_full(), false,  "is_full after reset"))       return;
    if (!check_u32(rr_history_get_overwrites(), 0, "overwrites after reset"))   return;

    rr_sample_t scratch[RR_HISTORY_CAPACITY];
    const rr_sample_t *result = rr_history_get_linear(scratch);
    if (!check_null(result, "get_linear after reset")) return;

    /* Re-fill and verify it works correctly */
    rr_history_add(750, 5000);
    if (!check_u16(rr_history_size(), 1, "size after refill")) return;

    const rr_sample_t *refill_data = rr_history_get_linear(scratch);
    if (!check_not_null(refill_data, "get_linear after refill")) return;
    if (!check_u16(refill_data[0].rr_ms, 750, "refill rr")) return;
    if (!check_u32(refill_data[0].timestamp_ms, 5000, "refill ts")) return;

    test_pass();
}

/*
 * ==================================================================
 * TEST 11 — Size reporting
 *
 * Verify that rr_history_size() returns the correct count
 * at every stage from 0 to CAPACITY to overflow.
 * ==================================================================
 */
static void test_size_reporting(void)
{
    test_begin("Size reporting");

    rr_history_init();

    /* Check size grows correctly */
    for (uint16_t i = 1; i <= RR_HISTORY_CAPACITY; i++)
    {
        rr_history_add((uint16_t)(800 + (i % 50)), i * 1000);

        char label[64];
        snprintf(label, sizeof(label), "size at %u inserts", (unsigned)i);
        if (!check_u16(rr_history_size(), i, label)) return;
    }

    /* Check size stays at CAPACITY during overflow */
    for (uint16_t i = 0; i < 100; i++)
    {
        rr_history_add((uint16_t)(800 + i), (uint32_t)((RR_HISTORY_CAPACITY + i) * 1000));

        char label[64];
        snprintf(label, sizeof(label), "size at overflow %u", (unsigned)i);
        if (!check_u16(rr_history_size(), RR_HISTORY_CAPACITY, label)) return;
    }

    test_pass();
}

/*
 * ==================================================================
 * TEST 12 — Full flag
 *
 * Verify that rr_history_is_full() returns:
 *   - false for 0..CAPACITY-1 inserts
 *   - true once CAPACITY is reached
 *   - true during overflow
 * ==================================================================
 */
static void test_full_flag(void)
{
    test_begin("Full flag");

    rr_history_init();

    /* Not full until we reach capacity */
    for (uint16_t i = 1; i < RR_HISTORY_CAPACITY; i++)
    {
        rr_history_add((uint16_t)(800 + (i % 50)), i * 1000);
        if (rr_history_is_full())
        {
            char buf[64];
            snprintf(buf, sizeof(buf), "is_full true too early at %u inserts", (unsigned)i);
            test_fail(buf);
            return;
        }
    }

    /* Add the last entry to reach capacity */
    rr_history_add(999, (uint32_t)(RR_HISTORY_CAPACITY * 1000));
    if (!check_bool(rr_history_is_full(), true, "is_full at capacity")) return;

    /* Still full during overflow */
    for (uint16_t i = 0; i < 10; i++)
    {
        rr_history_add((uint16_t)(800 + i), (uint32_t)((RR_HISTORY_CAPACITY + 1 + i) * 1000));
        if (!check_bool(rr_history_is_full(), true, "is_full during overflow")) return;
    }

    /* After reset, should not be full */
    rr_history_reset();
    if (!check_bool(rr_history_is_full(), false, "is_full after reset")) return;

    test_pass();
}

/*
 * ==================================================================
 * TEST 13 — Stress test (100,000 insertions)
 *
 * Verify that:
 *   - no memory corruption occurs
 *   - buffer remains at CAPACITY after overflow
 *   - overwrite counter increments correctly
 *   - linearised data has correct count and plausibility
 * ==================================================================
 */
static void test_stress(void)
{
    test_begin("Stress test (100000 insertions)");

    rr_history_init();

    const uint32_t total_inserts = 100000;

    for (uint32_t i = 0; i < total_inserts; i++)
    {
        /*
         * Vary rr: 400..999 ms (simulating sleep HRV)
         * Increase timestamp by 600..1200 ms (simulating real RR intervals)
         */
        uint16_t rr_val  = (uint16_t)(400 + (i % 600));
        uint32_t ts_val  = (uint32_t)(i * 800 + (i % 5) * 50);

        rr_history_add(rr_val, ts_val);
    }

    /* Verify invariants */
    if (!check_u16(rr_history_size(), RR_HISTORY_CAPACITY, "size after stress"))     return;
    if (!check_bool(rr_history_is_full(), true,          "is_full after stress"))    return;
    if (!check_u32(rr_history_get_overwrites(),
                   (uint32_t)(total_inserts - RR_HISTORY_CAPACITY), "overwrites after stress")) return;

    /* Linearise and verify */
    rr_sample_t *scratch = (rr_sample_t *)malloc(RR_HISTORY_CAPACITY * sizeof(rr_sample_t));
    if (scratch == NULL)
    {
        test_fail("malloc failed");
        return;
    }

    const rr_sample_t *data = rr_history_get_linear(scratch);
    if (!check_not_null(data, "get_linear after stress"))
    {
        free(scratch);
        return;
    }

    /* Verify monotonic timestamps */
    uint32_t prev_ts = data[0].timestamp_ms;
    for (uint16_t i = 1; i < RR_HISTORY_CAPACITY; i++)
    {
        if (data[i].timestamp_ms <= prev_ts)
        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "timestamp non-monotonic at index %u: %u <= %u",
                     (unsigned)i, (unsigned)data[i].timestamp_ms, (unsigned)prev_ts);
            test_fail(buf);
            free(scratch);
            return;
        }
        prev_ts = data[i].timestamp_ms;
    }

    /* Verify timestamp span is plausible: should cover ~CAPACITY*800 ms */
    uint32_t span = data[RR_HISTORY_CAPACITY - 1].timestamp_ms - data[0].timestamp_ms;
    uint32_t expected_min = (uint32_t)(RR_HISTORY_CAPACITY - 1) * 600; /* minimum RR=600 */
    uint32_t expected_max = (uint32_t)(RR_HISTORY_CAPACITY - 1) * 1200; /* max RR=1200 */

    if (span < expected_min || span > expected_max)
    {
        char buf[128];
        snprintf(buf, sizeof(buf),
                 "timestamp span %u ms outside expected range [%u, %u]",
                 (unsigned)span, (unsigned)expected_min, (unsigned)expected_max);
        test_fail(buf);
        free(scratch);
        return;
    }

    /* Verify oldest surviving entry matches first non-evicted */
    uint32_t first_kept_idx = total_inserts - RR_HISTORY_CAPACITY;  /* = 100000 - 512 = 99488 */
    uint16_t expected_rr   = (uint16_t)(400 + (first_kept_idx % 600));
    uint32_t expected_ts   = (uint32_t)(first_kept_idx * 800 + (first_kept_idx % 5) * 50);

    if (!check_u16(data[0].rr_ms, expected_rr, "oldest rr after stress")) { free(scratch); return; }
    if (!check_u32(data[0].timestamp_ms, expected_ts, "oldest ts after stress")) { free(scratch); return; }

    free(scratch);
    test_pass();
}

/*
 * ==================================================================
 * SIZEOF VERIFICATION
 *
 * Print sizeof(rr_sample_t), alignment, padding, and total RAM usage.
 * ==================================================================
 */
static void print_sizeof_info(void)
{
    printf("\n");
    printf("========================================================\n");
    printf("Memory Layout: rr_sample_t\n");
    printf("========================================================\n");
    printf("  sizeof(rr_sample_t)       = %zu bytes\n", sizeof(rr_sample_t));
    printf("  sizeof(timestamp_ms)      = %zu bytes (%s)\n",
           sizeof(((rr_sample_t *)0)->timestamp_ms),
           sizeof(((rr_sample_t *)0)->timestamp_ms) == 4 ? "uint32_t" : "UNEXPECTED");
    printf("  sizeof(rr_ms)             = %zu bytes (%s)\n",
           sizeof(((rr_sample_t *)0)->rr_ms),
           sizeof(((rr_sample_t *)0)->rr_ms) == 2 ? "uint16_t" : "UNEXPECTED");
    printf("  offsetof(timestamp_ms)    = %zu\n", offsetof(rr_sample_t, timestamp_ms));
    printf("  offsetof(rr_ms)           = %zu\n", offsetof(rr_sample_t, rr_ms));
    printf("  padding bytes             = %zu\n",
           sizeof(rr_sample_t) - sizeof(uint32_t) - sizeof(uint16_t));
    printf("  alignment requirement     = %zu\n", __alignof__(rr_sample_t));
    printf("\n");
    printf("  Total buffer: %u x %zu = %zu bytes\n",
           RR_HISTORY_CAPACITY, sizeof(rr_sample_t),
           (size_t)RR_HISTORY_CAPACITY * sizeof(rr_sample_t));
    printf("  Static RAM usage          = %zu bytes\n",
           (size_t)RR_HISTORY_CAPACITY * sizeof(rr_sample_t));
    printf("\n");
    printf("  RR_HISTORY_CAPACITY       = %u\n", RR_HISTORY_CAPACITY);
    printf("  RR_HISTORY_MASK           = 0x%04X\n", RR_HISTORY_MASK);
    printf("  Power of 2?               = %s\n",
           (RR_HISTORY_CAPACITY & (RR_HISTORY_CAPACITY - 1)) == 0 ? "YES" : "NO");
    printf("========================================================\n");
    printf("\n");
}

/*
 * ==================================================================
 * Main
 * ==================================================================
 */
int main(void)
{
    printf("\n");
    printf("========================================================\n");
    printf(" rr_history.c — Unit Test Suite\n");
    printf("========================================================\n");
    printf("\n");

    print_sizeof_info();

    printf("Running %d tests...\n\n", 13);

    test_empty_buffer();
    test_single_insertion();
    test_multiple_insertions();
    test_buffer_reaches_capacity();
    test_buffer_overflow();
    test_wrap_around();
    test_linearization_order();
    test_timestamp_preservation();
    test_rr_preservation();
    test_reset();
    test_size_reporting();
    test_full_flag();
    test_stress();

    printf("\n");
    printf("========================================================\n");

    int total = g_tests_passed + g_tests_failed;

    if (g_tests_failed == 0)
    {
        printf(" RESULT: ALL %d TESTS PASSED\n", total);
    }
    else
    {
        printf(" RESULT: %d/%d PASSED, %d FAILED\n",
               g_tests_passed, total, g_tests_failed);
    }

    printf("========================================================\n");
    printf("\n");

    return (g_tests_failed > 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
