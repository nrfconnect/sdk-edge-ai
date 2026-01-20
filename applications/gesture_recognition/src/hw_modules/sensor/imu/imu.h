/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup imu Inertial Measurement Unit (IMU)
 * @{
 * @ingroup bsp
 *
 * @brief This module provides IMU sensor control functions.
 *
 */
#ifndef __SENSOR_IMU_H__
#define __SENSOR_IMU_H__

#include <common.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Accelerometer full scale variants */
#define IMU_ACCEL_SCALE_2G      (2)
#define IMU_ACCEL_SCALE_4G      (4)
#define IMU_ACCEL_SCALE_8G      (8)
#define IMU_ACCEL_SCALE_16G     (16)

/** Gyroscope full scale variants */
#define IMU_ACCEL_SCALE_125DPS  (125)
#define IMU_ACCEL_SCALE_250DPS  (250)
#define IMU_ACCEL_SCALE_500DPS  (500)
#define IMU_ACCEL_SCALE_1000DPS (1000)
#define IMU_ACCEL_SCALE_2000DPS (2000)

/**
 * @brief IMU sensor configurations
 */
typedef struct imu_config_s
{
	/** Accelerometer full scale in G */
	int32_t accel_fs_g;

	 /** Gyroscope full scale in DPS */
	int32_t gyro_fs_dps;

	/** IMU data rate in Hz */
	int32_t data_rate_hz;
} imu_config_t;

/** Inertial sensor data */
typedef struct imu_data_s
{
	/** Accelerometer data */
	struct
	{
		int16_t raw;
		float phys;
	} accel[3];
	/** Gyroscope data */
	struct
	{
		int16_t raw;
		float phys;
	} gyro[3];
} imu_data_t;

/**
 * @brief Initialize and start generation of IMU sensor data
 * 
 * @param p_config          IMU configuration settings @ref imu_config_t
 * @param data_ready_cb     Data ready callback, provided callback will be
 *                          called when new data sample is ready for reading
 * 
 * @return Operation status @ref status_t 
 */
status_t imu_init(const imu_config_t *p_config,
			  generic_cb_t data_ready_cb);

/**
 * @brief Read IMU sensor data 
 * 
 * @param p_data        Pointer to data to be filled @ref imu_data_t
 *  
 * @return Operation status @ref status_t 
 */
status_t imu_read(imu_data_t *const p_data);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __SENSOR_IMU_H__ */

/**
 * @}
 */
