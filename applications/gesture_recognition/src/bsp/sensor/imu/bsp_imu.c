#include "bsp_imu.h"

#include <zephyr/types.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/sensor.h>

//////////////////////////////////////////////////////////////////////////////

static struct 
{
    bool initialized;
    bsp_generic_cb_t data_ready_cb;
    const struct device* dev;
} imu_ctx_ = {0};

//////////////////////////////////////////////////////////////////////////////

static void data_read_timer_handler(struct k_timer *timer)
{
    (void) timer;
    if (imu_ctx_.data_ready_cb)
    {
        imu_ctx_.data_ready_cb();
    }
}

K_TIMER_DEFINE(data_ready_timer_, data_read_timer_handler, NULL);

//////////////////////////////////////////////////////////////////////////////

bsp_status_t bsp_imu_init(const bsp_imu_config_t* p_config,
                            bsp_generic_cb_t data_ready_cb)
{
    BSP_NULL_CHECK(p_config);
 
    if (imu_ctx_.dev == NULL)
        imu_ctx_.dev = DEVICE_DT_GET_ONE(bosch_bmi270);

    BSP_RETURN_IF(imu_ctx_.dev == NULL, BSP_STATUS_HARDWARE_ERROR);

    int res = 0;
    struct sensor_value full_scale = {0};
    struct sensor_value sampling_freq = {0};
    struct sensor_value oversampling = {0};

    /* Setting scale in G, due to loss of precision if the SI unit m/s^2
	 * is used
	 */
	full_scale.val1 = p_config->accel_fs_g;            /* G */
	full_scale.val2 = 0;
	sampling_freq.val1 = p_config->data_rate_hz;       /* Hz. Performance mode */
	sampling_freq.val2 = 0;
	oversampling.val1 = 1;          /* Normal mode */
	oversampling.val2 = 0;

    res = sensor_attr_set(imu_ctx_.dev, SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_FULL_SCALE, &full_scale);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

	res = sensor_attr_set(imu_ctx_.dev, SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_OVERSAMPLING, &oversampling);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

    res = sensor_attr_set(imu_ctx_.dev, SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

    /* Setting scale in degrees/s to match the sensor scale */
	full_scale.val1 = p_config->gyro_fs_dps;          /* dps */
	full_scale.val2 = 0;
	sampling_freq.val1 = p_config->data_rate_hz;       /* Hz. Performance mode */
	sampling_freq.val2 = 0;
	oversampling.val1 = 1;          /* Normal mode */
	oversampling.val2 = 0;

	res = sensor_attr_set(imu_ctx_.dev, SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_FULL_SCALE, &full_scale);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

	res = sensor_attr_set(imu_ctx_.dev, SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_OVERSAMPLING, &oversampling);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

    imu_ctx_.data_ready_cb = data_ready_cb;
    const uint32_t data_ready_timer_period = 1000 / p_config->data_rate_hz;
    k_timer_start(&data_ready_timer_, K_MSEC(data_ready_timer_period), K_MSEC(data_ready_timer_period));    

	/* Set sampling frequency last as this also sets the appropriate
	 * power mode. If already sampling, change sampling frequency to
	 * 0.0Hz before changing other attributes
	 */
	res = sensor_attr_set(imu_ctx_.dev, SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

    return BSP_STATUS_SUCCESS;
}
    
//////////////////////////////////////////////////////////////////////////////

bsp_status_t bsp_imu_read(bsp_imu_data_t* const p_data)
{
    BSP_NULL_CHECK(p_data);
    BSP_RETURN_IF(imu_ctx_.dev == NULL, BSP_STATUS_HARDWARE_ERROR);

    struct sensor_value acc[3], gyr[3];

    int res = sensor_sample_fetch(imu_ctx_.dev);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

    res = sensor_channel_get(imu_ctx_.dev, SENSOR_CHAN_ACCEL_XYZ, acc);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

    res = sensor_channel_get(imu_ctx_.dev, SENSOR_CHAN_GYRO_XYZ, gyr);
    BSP_RETURN_IF(res != 0, BSP_STATUS_HARDWARE_ERROR);

    for (int i = 0; i < 3; i++)
    {
        p_data->accel[i].phys = (float)sensor_value_to_double(&acc[i]);
        p_data->accel[i].raw = (p_data->accel[i].phys * 1000);

        p_data->gyro[i].phys = (float)sensor_value_to_double(&gyr[i]);
        p_data->gyro[i].raw = (p_data->gyro[i].phys * 1000);
    }

    return BSP_STATUS_SUCCESS;
}