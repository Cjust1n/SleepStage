#include "mpu6050.h"

#include "hx_drv_iic.h"
#include "timer_interface.h"
#include "xprintf.h"

IIC_ERR_CODE_E mpu6050_i2c_write(uint8_t reg, uint8_t *data, uint8_t len)
{
    uint8_t addr[1];
    addr[0] = reg;

    return hx_drv_i2cm_write_data(
        USE_DW_IIC_0,
        MPU6050_I2C_ADDR,
        addr,
        1,
        data,
        len);
}

IIC_ERR_CODE_E mpu6050_i2c_read(uint8_t reg, uint8_t *data, uint8_t len)
{
    uint8_t addr[1];
    addr[0] = reg;

    return hx_drv_i2cm_write_restart_read(
        USE_DW_IIC_0,
        MPU6050_I2C_ADDR,
        addr,
        1,
        data,
        len);
}

IIC_ERR_CODE_E mpu6050_init(void)
{
    IIC_ERR_CODE_E err = IIC_ERR_OK;

    uint8_t data;

    xprintf("MPU6050 Init\r\n");

    /* Wake up MPU6050 */
    data = 0x01;
    err |= mpu6050_i2c_write(MPU6050_REG_PWR_MGMT_1, &data, 1);

    hx_drv_timer_cm55x_delay_ms(100, TIMER_STATE_DC);for (volatile uint32_t i = 0; i < 400000; i++)
    {
        __NOP();
    }
    /* Check WHO_AM_I */
    err |= mpu6050_i2c_read(MPU6050_REG_WHO_AM_I, &data, 1);

    xprintf("WHO_AM_I = 0x%02X\r\n", data);

    if(data != MPU6050_WHO_AM_I_VALUE)
    {
        xprintf("MPU6050 NOT FOUND!\r\n");
        return IIC_ERR_SYS;
    }

    /* Sample Rate = 100Hz
       Fs = 1kHz/(1+9)=100Hz */
    data = 19;
    err |= mpu6050_i2c_write(MPU6050_REG_SMPLRT_DIV, &data, 1);

    /* DLPF = 42Hz */
    data = 3;
    err |= mpu6050_i2c_write(MPU6050_REG_CONFIG, &data, 1);

    /* Gyro ±250 dps (not used) */
    data = 0x00;
    err |= mpu6050_i2c_write(MPU6050_REG_GYRO_CONFIG, &data, 1);

    /* Accel ±2g */
    data = 0x00;
    err |= mpu6050_i2c_write(MPU6050_REG_ACCEL_CONFIG, &data, 1);

    hx_drv_timer_cm55x_delay_ms(100, TIMER_STATE_DC);

    return err;
}

IIC_ERR_CODE_E mpu6050_get_accel_axis(mpu6050_axis_t *axis)
{
    uint8_t buf[6];

    IIC_ERR_CODE_E err;

    err = mpu6050_i2c_read(
            MPU6050_REG_ACCEL_XOUT_H,
            buf,
            6);

    if(err != IIC_ERR_OK)
        return err;

    axis->x = (int16_t)((buf[0] << 8) | buf[1]);
    axis->y = (int16_t)((buf[2] << 8) | buf[3]);
    axis->z = (int16_t)((buf[4] << 8) | buf[5]);

    return err;
}