/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup uxsm UX State Machine
 * @{
 * @ingroup app_gr
 *
 * uxsm means ux_state_machine.
 *
 * @brief Tracks the current operating state of the application and routes
 *        button clicks to the corresponding action.
 *
 * Two orthogonal pieces of state are owned here:
 *
 *   - @ref ux_btn_func_t        : NORMAL vs. PAIRING (determines button press actions)
 *   - @ref ux_remotectrl_mode_t : PRESENTATION vs. MUSIC keyboard control mapping
 *
 * Click dispatch:
 *
 *   - UX_BTN_FUNC_NORMAL:
 *     * SHORT click -> toggle remote-control mode (PRESENTATION <-> MUSIC)
 *     * LONG  click -> forget bonded devices (only when BLE HID is enabled)
 *
 *   - UX_BTN_FUNC_PAIRING:
 *     * SHORT click -> reject pending pairing
 *     * LONG  click -> confirm pending pairing
 */
#ifndef __UXSM_H__
#define __UXSM_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Button function mode: determines what button presses do.
 */
typedef enum {
	UX_BTN_FUNC_NORMAL = 0,
	UX_BTN_FUNC_PAIRING,
} ux_btn_func_t;

/**
 * @brief Remote-control mode: which key mapping is active for recognized gestures.
 */
typedef enum {
	UX_REMOTECTRL_MODE_PRESENTATION = 0,
	UX_REMOTECTRL_MODE_MUSIC,
} ux_remotectrl_mode_t;

/**
 * @brief Initialize the UX state manager.
 *
 * Registers the click handler with the button module. Must be called after
 * @ref button_init().
 *
 * @return 0 on success, negative error code otherwise.
 */
int uxsm_init(void);

/**
 * @brief Set the button function mode.
 *
 * Logs the transition and prints user-facing button instructions for the new mode.
 */
void uxsm_set_btn_func(ux_btn_func_t btn_func);

/**
 * @brief Get the current button function mode.
 */
ux_btn_func_t uxsm_get_btn_func(void);

/**
 * @brief Get the current remote-control mode.
 */
ux_remotectrl_mode_t uxsm_get_remotectrl_mode(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __UXSM_H__ */

/**
 * @}
 */
