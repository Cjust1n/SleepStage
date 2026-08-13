/*
 * xprintf.h — Host-side stub for embedded xprintf
 *
 * Maps the embedded xprintf() to standard printf() so that
 * rr_history.c can be compiled and tested on a Linux desktop.
 *
 * Usage:
 *   gcc test_rr_history.c ../rr_history.c -I. -o test_rr_history
 */

#ifndef XPRINTF_STUB_H
#define XPRINTF_STUB_H

#include <stdio.h>

/*
 * Map embedded xprintf to standard printf.
 * Carriage returns (\r) are harmless on Linux — they just
 * move the cursor to column 0, which is fine for test output.
 */
#define xprintf(...)  printf(__VA_ARGS__)

#endif /* XPRINTF_STUB_H */
