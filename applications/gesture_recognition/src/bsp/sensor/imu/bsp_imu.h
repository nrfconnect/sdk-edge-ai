/**
 *
 * @defgroup bsp_imu Inertial Measurement Unit (IMU)
 * @{
 * @ingroup bsp
 *
 * @brief This module provides IMU sensor control functions.
 *
 */
#ifndef __BSP_SENSOR_IMU_H__
#define __BSP_SENSOR_IMU_H__

#include <bsp_common.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Accelerometer full scale variants */
#define BSP_IMU_ACCEL_SCALE_2G      (2)
#define BSP_IMU_ACCEL_SCALE_4G      (4)
#define BSP_IMU_ACCEL_SCALE_8G      (8)
#define BSP_IMU_ACCEL_SCALE_16G     (16)

/** Gyroscope full scale variants */
#define BSP_IMU_ACCEL_SCALE_125DPS  (125)
#define BSP_IMU_ACCEL_SCALE_250DPS  (250)
#define BSP_IMU_ACCEL_SCALE_500DPS  (500)
#define BSP_IMU_ACCEL_SCALE_1000DPS (1000)
#define BSP_IMU_ACCEL_SCALE_2000DPS (2000)

/**
 * @brief IMU sensor configurations
 */
typedef struct bsp_imu_config_s
{
    /** Accelerometer full scale in G */
    int32_t accel_fs_g;

     /** Gyroscope full scale in DPS */
    int32_t gyro_fs_dps;

    /** IMU data rate in Hz */
    int32_t data_rate_hz;
} bsp_imu_config_t;

/** Inertial sensor data */
typedef struct bsp_imu_data_s
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
} bsp_imu_data_t;

/**
 * @brief Initialize and start generation of IMU sensor data
 * 
 * @param p_config          IMU configuration settings @ref bsp_imu_config_t
 * @param data_ready_cb     Data ready callback, provided callback will be called when new data sample is ready for reading
 * 
 * @return Operation status @ref bsp_status_t 
 */
bsp_status_t bsp_imu_init(const bsp_imu_config_t* p_config,
                            bsp_generic_cb_t data_ready_cb);

/**
 * @brief Read IMU sensor data 
 * 
 * @param p_data        Pointer to data to be filled @ref bsp_imu_data_t
 *  
 * @return Operation status @ref bsp_status_t 
 */
bsp_status_t bsp_imu_read(bsp_imu_data_t* const p_data);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __BSP_SENSOR_IMU_H__ */

/**
 * @}
 */
