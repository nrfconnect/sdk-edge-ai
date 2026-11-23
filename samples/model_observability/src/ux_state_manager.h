/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup sample_mo Model Observability Sample
 *
 * @defgroup sample_mo_uxsm UX State Machine
 * @{
 * @ingroup sample_mo
 *
 * uxsm means ux_state_machine.
 *
 * @brief Tracks the BLE connection state of the sample and drives
 *        the blue status LED accordingly.
 *
 * LED indication (LED ON time is @ref UX_LED_ON_MS in both states):
 *
 *   - @ref UX_STATE_ADVERTISING : blink at 3 Hz (waiting for BLE connection)
 *   - @ref UX_STATE_CONNECTED   : blink at 1 Hz (BLE peer connected)
 */
#ifndef __UXSM_H__
#define __UXSM_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Status LED ON time per blink, in milliseconds */
#define UX_LED_ON_MS (100U)

/**
 * @brief Sample UX state.
 */
typedef enum {
	UX_STATE_ADVERTISING = 0,
	UX_STATE_CONNECTED,
} ux_state_t;

/**
 * @brief Initialize the UX state manager.
 *
 * Initializes the status LED and starts blinking in @ref UX_STATE_ADVERTISING.
 *
 * @return 0 on success, negative error code otherwise.
 */
int uxsm_init(void);

/**
 * @brief Update the state from the BLE connection state.
 *
 * Safe to call from BLE stack callbacks. Matches @ref ble_mds_conn_cb_t.
 *
 * @param connected true if a peer is connected, false otherwise.
 */
void uxsm_set_ble_connected(bool connected);

/**
 * @brief Get the current UX state.
 */
ux_state_t uxsm_get_state(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __UXSM_H__ */

/**
 * @}
 */
