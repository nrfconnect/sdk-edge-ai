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
extern "C"
{
#endif /* __cplusplus */


/**
 * @brief BLE connection callback, this callback will be called when state of the connection is changed
 *
 * @param connected     BLE connected state, true if connected, otherwise false
 */
typedef void (*ble_connection_cb_t)(bool connected);


#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __BLE_COMMON_H__ */

/**
 * @}
 */
