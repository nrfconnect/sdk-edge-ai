/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file ble_nus.h
 * @defgroup ble_nus Bluetooth NUS interface
 * @{
 * @ingroup ble
 *
 * Nordic UART Service interface for streaming raw IMU data over BLE.
 * Used in data collection mode only.
 */

#ifndef __BLE_NUS_H__
#define __BLE_NUS_H__

#include "../ble_common.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief Initialize BLE NUS and start advertising
 *
 * @param cb Connection callback
 *
 * @return 0 on success, negative errno on failure
 */
int ble_nus_init(ble_connection_cb_t cb);

/**
 * @brief Send raw IMU sample data over NUS
 *
 * @param input_data Array of 6 int16 values: accel_x,y,z, gyro_x,y,z
 *
 * @return 0 on success, negative errno on failure
 */
int ble_nus_send(const int16_t *input_data);

#ifdef __cplusplus
}
#endif

#endif /* __BLE_NUS_H__ */

/**
 * @}
 */
