/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup accel Accelerometer
 * @{
 * @ingroup app_mo
 *
 * @brief BMI270 accelerometer control functions.
 *
 * Sensor settings and sample conversion match the data_forwarder sample
 * (samples/data_forwarder, BMI270 build) used to collect the training dataset,
 * so the model sees the same data on device as during training.
 */
#ifndef __SENSOR_ACCEL_H__
#define __SENSOR_ACCEL_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Accelerometer full scale in G, as in data_forwarder */
#define ACCEL_SCALE_G (4)

/** Accelerometer sampling rate in Hz, as in data_forwarder */
#define ACCEL_SAMPLE_RATE_HZ (100U)

/** Number of accelerometer axes */
#define ACCEL_AXES_NUM (3U)

/** Data ready callback, called from timer (ISR) context every sample period */
typedef void (*accel_data_ready_cb_t)(void);

/**
 * @brief Initialize the accelerometer and start generation of data ready events
 *
 * @param data_ready_cb Data ready callback, can be NULL
 *
 * @retval 0 Success.
 * @retval -ENODEV Sensor device is not ready.
 * @retval -errno Negative error code from the sensor driver.
 */
int accel_init(accel_data_ready_cb_t data_ready_cb);

/**
 * @brief Read one accelerometer sample
 *
 * Fills @p p_data as ax, ay, az in m/s^2 (Zephyr sensor units), converted
 * with sensor_value_to_float() as in data_forwarder.
 *
 * @param p_data Output buffer of @ref ACCEL_AXES_NUM values
 *
 * @retval 0 Success.
 * @retval -EINVAL @p p_data is NULL.
 * @retval -errno Negative error code from the sensor driver.
 */
int accel_read(float p_data[ACCEL_AXES_NUM]);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __SENSOR_ACCEL_H__ */

/**
 * @}
 */
