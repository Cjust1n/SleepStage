/*
 * max30102.c — MAX30102 HR-only driver
 *
 * Uses same I2C bus as MPU6050 (USE_DW_IIC_0).
 * Assumes MPU6050 driver has already initialized the bus.
 */
#include "max30102.h"
#include "xprintf.h"
#include <stdlib.h>

// I2C write helper
static IIC_ERR_CODE_E max_write(uint8_t reg, uint8_t val) {
    uint8_t addr[1];
    addr[0] = reg;
    return hx_drv_i2cm_write_data(USE_DW_IIC_0, MAX30102_I2C_ADDR, addr, 1, &val, 1);
}

// I2C read helper using discrete write (register) then read transactions
static IIC_ERR_CODE_E max_read(uint8_t reg, uint8_t *data, uint8_t len) {
    IIC_ERR_CODE_E err;
    uint8_t addr[1];
    addr[0] = reg;

    // 1. Write the target register address
    err = hx_drv_i2cm_write_data(USE_DW_IIC_0, MAX30102_I2C_ADDR, addr, 1, NULL, 0);
    if (err != IIC_ERR_OK) {
        return err;
    }

    // 2. Read the data back from that register
    return hx_drv_i2cm_read_data(USE_DW_IIC_0, MAX30102_I2C_ADDR, data, len);
}

int max30102_init(void) {
    uint8_t tmp;

    // Note: I2C bus assumed already initialized by mpu6050_init().
    // If MPU6050 not initialized first, uncomment the line below:
    // hx_drv_i2cm_init(USE_DW_IIC_0, HX_I2C_HOST_MST_0_BASE, DW_IIC_SPEED_FAST);

    // 1. Verify PART_ID (should be 0x15)
    if (max_read(MAX30102_PART_ID, &tmp, 1) != IIC_ERR_OK) {
        xprintf("MAX30102 read PART_ID fail\n");
        return -1;
    }
    if (tmp != MAX30102_PART_ID_VALUE) {
        xprintf("MAX30102 PART_ID mismatch: 0x%02X (expect 0x15)\n", tmp);
        return -1;
    }
    xprintf("MAX30102 PART_ID OK: 0x%02X\n", tmp);

    // 2. Soft reset
    if (max_write(MAX30102_MODE_CONFIG, MAX30102_MODE_RESET) != IIC_ERR_OK) {
        xprintf("MAX30102 reset fail\n");
        return -1;
    }
    // Wait for reset to complete (reset bit auto-clears)
    for (volatile int d = 0; d < 100000; ++d);

    // 3. Set MODE_CONFIG = HR-only (Red LED)
    if (max_write(MAX30102_MODE_CONFIG, MAX30102_MODE_HR_ONLY) != IIC_ERR_OK) {
        xprintf("MAX30102 mode config fail\n");
        return -1;
    }

    // 4. Set SPO2_CONFIG (ADC range, sample rate, pulse width)
    if (max_write(MAX30102_SPO2_CONFIG, MAX30102_SPO2_CFG) != IIC_ERR_OK) {
        xprintf("MAX30102 spo2 config fail\n");
        return -1;
    }

    // 5. Set FIFO_CONFIG (averaging, rollover)
    if (max_write(MAX30102_FIFO_CONFIG, MAX30102_FIFO_CFG) != IIC_ERR_OK) {
        xprintf("MAX30102 fifo config fail\n");
        return -1;
    }

    // 6. Set LED current (low, ~6.2 mA)
    if (max_write(MAX30102_LED1_PA, MAX30102_LED_CURRENT_LOW) != IIC_ERR_OK) {
        xprintf("MAX30102 LED1 current fail\n");
        return -1;
    }

    // 7. Clear FIFO pointers
    max_write(MAX30102_FIFO_WR_PTR, 0);
    max_write(MAX30102_OVF_COUNTER, 0);
    max_write(MAX30102_FIFO_RD_PTR, 0);

    xprintf("MAX30102 init done\n");
    return 0;
}

/******************************************************************************
 * Get number of unread samples in FIFO
 ******************************************************************************/
int max30102_available_samples(uint8_t *out_count)
{
    uint8_t wr_ptr;
    uint8_t rd_ptr;

    if (out_count == NULL)
        return -1;

    if (max_read(MAX30102_FIFO_WR_PTR, &wr_ptr, 1) != IIC_ERR_OK)
        return -1;

    if (max_read(MAX30102_FIFO_RD_PTR, &rd_ptr, 1) != IIC_ERR_OK)
        return -1;

    // FIFO pointer is 5-bit (0~31)
    *out_count = (wr_ptr - rd_ptr) & 0x1F;

    return 0;
}

/******************************************************************************
 * Read any MAX30102 register (for debug/verification)
 ******************************************************************************/
int max30102_read_register(uint8_t reg, uint8_t *val)
{
    if (val == NULL)
        return -1;
    return (max_read(reg, val, 1) == IIC_ERR_OK) ? 0 : -1;
}

/******************************************************************************
 * Read one raw PPG sample (Red LED, 18-bit)
 ******************************************************************************/
int max30102_read_hr_sample(uint32_t *out_red)
{
    uint8_t raw[3];

    if (out_red == NULL)
        return -1;

    // Read one FIFO sample (3 bytes)
    if (max_read(MAX30102_FIFO_DATA, raw, 3) != IIC_ERR_OK)
        return -1;

    *out_red =
        (((uint32_t)raw[0] << 16) |
         ((uint32_t)raw[1] << 8)  |
          (uint32_t)raw[2]) & 0x3FFFF;

    return 0;
}