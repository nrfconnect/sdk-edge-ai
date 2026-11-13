/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file ble_nus.h
 * @defgroup ble_nus Bluetooth NUS interface
 * @{
 * @ingroup app_mo
 *
 * Nordic UART Service interface for sending recognized class and
 * its probability over BLE.
 */

#ifndef __BLE_NUS_H__
#define __BLE_NUS_H__

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief BLE connection state change callback
 *
 * @param connected true if a peer is connected, false otherwise
 */
typedef void (*ble_nus_conn_cb_t)(bool connected);

/**
 * @brief Initialize BLE NUS and start advertising
 *
 * @param conn_cb Callback called on connection state change, can be NULL
 *
 * @return 0 on success, negative errno on failure
 */
int ble_nus_init(ble_nus_conn_cb_t conn_cb);

/**
 * @brief Send inference result over NUS as "<class>,<probability>\r\n" text line
 *
 * @param class_id        Predicted class index
 * @param probability_pct Probability of the predicted class in percent (0 - 100)
 *
 * @return 0 on success, -ENOTCONN if no peer subscribed, negative errno on failure
 */
int ble_nus_send_result(uint16_t class_id, uint8_t probability_pct);

#ifdef __cplusplus
}
#endif

#endif /* __BLE_NUS_H__ */

/**
 * @}
 */
