/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file ble_mds.h
 * @defgroup ble_mds Bluetooth MDS interface
 * @{
 * @ingroup app_mo
 *
 * Memfault Diagnostic Service interface. A bonded phone (nRF Memfault app)
 * drains the staged observability CDR and forwards it to the Memfault cloud.
 */

#ifndef __BLE_MDS_H__
#define __BLE_MDS_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief BLE connection state change callback
 *
 * @param connected true if a peer is connected, false otherwise
 */
typedef void (*ble_mds_conn_cb_t)(bool connected);

/**
 * @brief Initialize BLE, load bonds and start MDS advertising
 *
 * @param conn_cb Callback called on connection state change, can be NULL
 *
 * @return 0 on success, negative errno on failure
 */
int ble_mds_init(ble_mds_conn_cb_t conn_cb);

#ifdef __cplusplus
}
#endif

#endif /* __BLE_MDS_H__ */

/**
 * @}
 */
