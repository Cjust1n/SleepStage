/******************************************************************************
 * Minimal TFLM MobileNet Classification Application
 *
 * No Camera
 * No Sensor DP
 * No Event
 * No Power Mode
 * Pure Golden Reference Validation
 ******************************************************************************/

#include <stdint.h>

#include "WE2_device.h"
#include "board.h"
#include "xprintf.h"

#include "memory_manage.h"

#include "spi_eeprom_comm.h"
#include "common_config.h"

#include "cvapp_mb_cls.h"
#include "tflm_mb_cls.h"

/******************************************************************************
 * Main Application
 ******************************************************************************/


#define SLEEPSTAGE_MODEL_FLASH_ADDR 0x3AB7B000


int tflm_mb_cls_app(void)
{
    xprintf("\n");
    xprintf("=====================================\n");
    xprintf(" SLEEPSTAGE_MODEL Demo\n");
    xprintf(" Arm Cortex-M55 + Ethos-U55\n");
    xprintf("=====================================\n");

    /*
     * Enable SPI Flash XIP
     */

    hx_lib_spi_eeprom_open(USE_DW_SPI_MST_Q);

    hx_lib_spi_eeprom_enable_XIP(
        USE_DW_SPI_MST_Q,
        true,
        FLASH_QUAD,
        true);

#ifdef __GNU__

    extern char __mm_start_addr__;

    mm_set_initial(
        (int)&__mm_start_addr__,
        0x00200000 -
        ((int)&__mm_start_addr__ - 0x34000000));

#else

    static uint8_t mm_start_addr
        __attribute__((section(".bss.mm_start_addr")));

    mm_set_initial(
        (int)&mm_start_addr,
        0x00200000 -
        ((int)&mm_start_addr - 0x34000000));

#endif

    /*
     * Initialize TFLM
     */

    if (cv_mb_cls_init(
            true,
            true,
            SLEEPSTAGE_MODEL_FLASH_ADDR) != 0)
    {
        xprintf("Initialization FAILED\n");
        return -1;
    }

    /*
     * Run validation
     */
	struct_yolov8_ob_algoResult dummy;
    cv_mb_cls_run(&dummy);

    xprintf("\nValidation Complete\n");

    while (1)
    {
    }

    return 0;
}