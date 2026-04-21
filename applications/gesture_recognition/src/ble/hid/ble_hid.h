/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup ble_hid Bluetooth HID interface
 * @{
 * @ingroup ble
 *
 *
 */
#ifndef __BLE_HID_H__
#define __BLE_HID_H__

#include "../ble_common.h"
#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Supported HID keys to emulate keyboard
 */
typedef enum {
	BLE_HID_KEY_ARROW_LEFT = 0,
	BLE_HID_KEY_ARROW_RIGHT,
	BLE_HID_KEY_F5,
	BLE_HID_KEY_ESC,
	BLE_HID_KEY_MEDIA_PREV_TRACK,
	BLE_HID_KEY_MEDIA_NEXT_TRACK,
	BLE_HID_KEY_MEDIA_PLAY_PAUSE,
	BLE_HID_KEY_MEDIA_MUTE,
	BLE_HID_KEY_MEDIA_VOLUME_UP,
	BLE_HID_KEY_MEDIA_VOLUME_DOWN,

	BLE_HID_KEYS_count
} ble_hid_key_t;

/**
 * @brief Initialize BLE HID profile and start advertising.
 *
 * Registers authentication callbacks, initializes the HID service, and
 * enables the Bluetooth stack. Advertising is started asynchronously once
 * the stack is ready.
 *
 * @retval 0        Success.
 * @retval <0       Negative errno from the Bluetooth stack
 *                  (e.g. from bt_conn_auth_cb_register(), bt_hids_init(),
 *                  or bt_enable()).
 */
int ble_hid_init(void);

/**
 * @brief Send a keyboard or consumer-control key via the HID profile
 *
 * Sends a key press followed by a key release for the specified key.
 * Standard keys (arrows, F5, ESC) are sent as keyboard reports; media keys
 * (play/pause, volume, track, mute) are sent as consumer-control reports.
 *
 * @param key  Key to send, see @ref ble_hid_key_t
 *
 * @retval 0         Key sent successfully.
 * @retval -EBUSY    Device is currently in pairing mode; key not sent.
 * @retval -ENOTCONN No active BLE connection.
 * @retval -EAGAIN   BLE link is up but the peer has not yet enabled HID
 *                   notifications (CCCD not written); key not sent.
 * @retval -EINVAL   Unknown or unsupported key code.
 * @retval <0        Other negative errno from the underlying HID service.
 */
int ble_hid_send_key(ble_hid_key_t key);

/**
 * @brief Accept or reject a pending pairing confirmation request.
 *
 * Called from the UX state machine when the user acknowledges (or rejects) a
 * passkey confirmation request triggered by the remote peer during pairing
 * with MITM protection. This is only relevant when
 * @c CONFIG_BLE_MITM_AUTH is enabled. If no confirmation request is currently
 * pending, the function returns an error from the message queue API.
 *
 * @param accept  true to confirm the pairing, false to reject it.
 * @return Operation status, 0 for success.
 * @retval -ENOMSG No pairing confirmation request is pending.
 * @retval <0      Other negative errno from the Bluetooth stack.
 */
int ble_hid_confirm_pairing(bool accept);

/**
 * @brief Forget all bonded devices and disconnect all active connections.
 *
 * Removes all stored bonding information and disconnects any currently
 * active BLE connections. Advertising will be restarted automatically
 * by the disconnect handler.
 *
 * @return 0 on success, negative error code otherwise.
 */
int ble_hid_forget_bonds(void);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __BLE_HID_H__ */

/**
 * @}
 */
