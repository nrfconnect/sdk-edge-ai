/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup ble_gatt Bluetooth GATT interface
 * @{
 * @ingroup ble
 *
 *
 */
#ifndef __BLE_GATT_H__
#define __BLE_GATT_H__

#include "../ble_common.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */


/**
 * @brief BLE data received callback, this callback will be called when new data is received from the BLE GATT profile
 *
 * @param data          Received data
 * @param len           Length of the received data
 */
typedef void (*ble_data_received_cb_t)(const char *data, size_t len);

/**
 * @brief Initialize BLE GATT profile and start advertising
 *
 * @param connection_cb     Connection callback @ref ble_connection_cb_t
 * @param data_received_cb  Data received callback @ref ble_data_received_cb_t
 *
 * @return Operation status, 0 for success
 */
int ble_gatt_init(ble_connection_cb_t connection_cb, ble_data_received_cb_t data_received_cb);

/**
 * @brief Send Neuton inference result over BLE GATT profile
 *
 * @param data      BLE data to send
 * @param len       Data size in bytes
 *
 * @return Operation status, 0 for success
 */
int ble_gatt_send_raw_data(const uint8_t *data, size_t len);


/**
 * @brief Start BLE advertising
 *
 * @return Operation status, 0 for success
 */
int ble_gatt_start_advertising(void);

/**
 * @brief Get RSSI of the current connection
 *
 * @param out_rssi  Pointer to store RSSI value, can be NULL
 *
 * @return Operation status, 0 for success
 */

int ble_gatt_get_rssi(int8_t *out_rssi);


#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __BLE_GATT_H__ */

/**
 * @}
 */
