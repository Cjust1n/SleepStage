#ifndef MPU6050_H_
#define MPU6050_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "hx_drv_iic.h"
#include <stdint.h>

#define MPU6050_I2C_ADDR         0x68

typedef struct
{
    int16_t x;
    int16_t y;
    int16_t z;
} mpu6050_axis_t;

/* Registers */
#define MPU6050_REG_WHO_AM_I         0x75
#define MPU6050_REG_PWR_MGMT_1       0x6B
#define MPU6050_REG_SMPLRT_DIV       0x19
#define MPU6050_REG_CONFIG           0x1A
#define MPU6050_REG_GYRO_CONFIG      0x1B
#define MPU6050_REG_ACCEL_CONFIG     0x1C
#define MPU6050_REG_ACCEL_XOUT_H     0x3B
#define MPU6050_REG_ACCEL_YOUT_H     0x3D
#define MPU6050_REG_ACCEL_ZOUT_H     0x3F

#define MPU6050_WHO_AM_I_VALUE       0x68

IIC_ERR_CODE_E mpu6050_i2c_write(uint8_t reg, uint8_t *data, uint8_t len);
IIC_ERR_CODE_E mpu6050_i2c_read(uint8_t reg, uint8_t *data, uint8_t len);

IIC_ERR_CODE_E mpu6050_init(void);
IIC_ERR_CODE_E mpu6050_get_accel_axis(mpu6050_axis_t *axis);

#ifdef __cplusplus
}
#endif

#endif