#include <stdio.h>
#include <stdint.h>

#include "xprintf.h"
#include "timer_interface.h"
#include "hx_drv_scu.h"
#include "hx_drv_iic.h"

#include "imu_read_app.h"
#include "mpu6050.h"

int app_main(void)
{
    IIC_ERR_CODE_E err;
    mpu6050_axis_t accel;

    xprintf("Start MPU6050 Read App\n");

    /* I2C Pinmux */
    hx_drv_scu_set_PA2_pinmux(SCU_PA2_PINMUX_I2C_M_SCL, 1);
    hx_drv_scu_set_PA3_pinmux(SCU_PA3_PINMUX_I2C_M_SDA, 1);

    /* I2C Init */
    hx_drv_i2cm_init(USE_DW_IIC_0,
                     HX_I2C_HOST_MST_0_BASE,
                     DW_IIC_SPEED_FAST);

    hx_drv_i2cm_set_err_cb(USE_DW_IIC_0, i2cm_0_err_cb);

    err = mpu6050_init();

    if(err != IIC_ERR_OK)
    {
        xprintf("MPU6050 Init Failed\n");
        return -1;
    }

    xprintf("MPU6050 Init OK\n");

    while(1)
    {
        err = mpu6050_get_accel_axis(&accel);

        if(err == IIC_ERR_OK)
        {
            xprintf("Accel X : %6d\r\n", accel.x);
            xprintf("Accel Y : %6d\r\n", accel.y);
            xprintf("Accel Z : %6d\r\n", accel.z);
            xprintf("--------------------------\r\n");
        }
        else
        {
            xprintf("Read Error (%d)\r\n", err);
        }

        /* 50Hz sampling */
        hx_drv_timer_cm55x_delay_ms(20, TIMER_STATE_DC);
    }

    return 0;
}

volatile uint32_t g_err_cb = 0;

void i2cm_0_err_cb(void *status)
{
    HX_DRV_DEV_IIC *iic_obj = status;
    HX_DRV_DEV_IIC_INFO *iic_info_ptr = &(iic_obj->iic_info);

    g_err_cb = 1;
    xprintf("[%s] err:%d\r\n",
            __FUNCTION__,
            iic_info_ptr->err_state);
}