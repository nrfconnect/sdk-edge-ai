/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/** @defgroup ble BLE interface for GR */

/**
 *
 * @defgroup ble_common Common BLE interface
 * @{
 * @ingroup ble
 *
 *
 */
#ifndef __BLE_COMMON_H__
#define __BLE_COMMON_H__

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Set the current BLE connection state.
 *
 * Intended to be called from the BLE stack's connected()/disconnected()
 * callbacks to keep the shared connection state in sync.
 *
 * When CONFIG_BLE_MODE_NONE is enabled this function is still safe to
 * call, but it becomes a no-op with respect to any BLE-stack specific
 * notification (only the internal state is updated).
 *
 * @param connected true if a peer is connected, false otherwise.
 */
void ble_common_set_connected(bool connected);

/**
 * @brief Get the current BLE connection state.
 *
 * Safe to call regardless of the selected BLE mode. When
 * CONFIG_BLE_MODE_NONE is enabled this will always return false unless
 * an external module explicitly sets the state.
 *
 * @return true if a peer is currently connected, false otherwise.
 */
bool ble_common_is_connected(void);

/**
 * @brief Initialize the common BLE module.
 *
 * Performs one-time initialization of the shared BLE state used by the
 * common BLE interface. This must be called once during system startup,
 * before any other ble_common_* API is used and before the BLE stack
 * starts delivering connection state changes.
 *
 * Safe to call regardless of the selected BLE mode (including
 * CONFIG_BLE_MODE_NONE).
 */
void ble_common_init(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __BLE_COMMON_H__ */

/**
 * @}
 */
