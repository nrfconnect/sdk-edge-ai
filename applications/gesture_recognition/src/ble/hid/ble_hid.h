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
extern "C"
{
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
 * @brief Initialize BLE HID profile and start advertasing
 *
 * @param cb        Connection callback @ref ble_connection_cb_t
 *
 * @return Operation status, 0 for success
 */
int ble_hid_init(ble_connection_cb_t cb);

/**
 * @brief Send keyboard key via HID profile
 *
 * @param key       Keyboard key @ref ble_hid_key_t
 *
 * @return Operation status, 0 for success
 */
int ble_hid_send_key(ble_hid_key_t key);

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __BLE_HID_H__ */

/**
 * @}
 */
