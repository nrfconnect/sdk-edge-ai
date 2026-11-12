/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup uxsm UX State Machine
 * @{
 * @ingroup app_mo
 *
 * uxsm means ux_state_machine.
 *
 * @brief Tracks the BLE connection state of the application and drives
 *        the status LED accordingly.
 *
 * LED indication (LED ON time is @ref UX_LED_ON_MS in both states):
 *
 *   - @ref UX_STATE_ADVERTISING : blink at 3 Hz (waiting for BLE connection)
 *   - @ref UX_STATE_CONNECTED   : blink at 1 Hz (BLE peer connected)
 *
 * LED color shows the recognized activity (see @ref uxsm_set_current_activity):
 * idle - blue (default), updown - red, circle - green, flip - red + green.
 */
#ifndef __UXSM_H__
#define __UXSM_H__

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** Status LED ON time per blink, in milliseconds */
#define UX_LED_ON_MS (100U)

/** Consecutive recognitions of a class needed to switch the LED color */
#define UX_ACTIVITY_CONFIRM_COUNT (3U)

/** Number of other-class samples tolerated inside a run before it is dropped */
#define UX_ACTIVITY_FLICKER_TOLERANCE (1U)

/**
 * @brief Application UX state.
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
 * Safe to call from BLE stack callbacks. Matches @ref ble_nus_conn_cb_t.
 *
 * @param connected true if a peer is connected, false otherwise.
 */
void uxsm_set_ble_connected(bool connected);

/**
 * @brief Report the class recognized by the model; updates the LED color.
 *
 * The color changes only after @ref UX_ACTIVITY_CONFIRM_COUNT recognitions of
 * the same class, where up to @ref UX_ACTIVITY_FLICKER_TOLERANCE sample of
 * another class inside the run is ignored. Call from a single thread.
 *
 * @param label Model class (nrf_edgeai_user_label_t). Out of range values are ignored.
 */
void uxsm_set_current_activity(uint16_t label);

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
