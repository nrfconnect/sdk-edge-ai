/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_hid.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/logging/log.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/hids.h>

LOG_MODULE_REGISTER(ble_hid);

/* HID keyboard usage codes */
#define KEY_ARROW_LEFT  0x50
#define KEY_ARROW_RIGHT 0x4F
#define KEY_F5          0x3E
#define KEY_ESC         0x29

/* Consumer control bit positions (1-byte bitmap) */
#define KEY_MEDIA_VOLUME_UP   BIT(0)
#define KEY_MEDIA_VOLUME_DOWN BIT(1)
#define KEY_MEDIA_MUTE        BIT(2)
#define KEY_MEDIA_PLAY_PAUSE  BIT(3)
#define KEY_MEDIA_PREV_TRACK  BIT(4)
#define KEY_MEDIA_NEXT_TRACK  BIT(5)

/* Input report indices within the report group */
#define INPUT_REP_KEYBOARD_IDX 0
#define INPUT_REP_CONSUMER_IDX 1

/* Report IDs matching the report map */
#define INPUT_REP_KEYBOARD_ID 0x01
#define INPUT_REP_CONSUMER_ID 0x02

/* Report sizes in bytes */
#define INPUT_REP_KEYBOARD_LEN 8
#define INPUT_REP_CONSUMER_LEN 1

/* Offset of the 'key' byte within the HID report array */
#define INPUT_REP_KEYBOARD_KEY_OFFSET 2
#define INPUT_REP_CONSUMER_KEY_OFFSET 0

#define BASE_USB_HID_SPEC_VERSION 0x0101

BT_HIDS_DEF(hids_obj, INPUT_REP_KEYBOARD_LEN, INPUT_REP_CONSUMER_LEN);

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID16_ALL,
				  BT_UUID_16_ENCODE(BT_UUID_HIDS_VAL),
				  BT_UUID_16_ENCODE(BT_UUID_BAS_VAL)),
	BT_DATA_BYTES(BT_DATA_GAP_APPEARANCE,
				  BT_BYTES_LIST_LE16(BT_APPEARANCE_HID_PRESENTATION_REMOTE)),
};

static const struct bt_data sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME,
		sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static const uint8_t report_map[] = {
	0x05, 0x01, /* Usage Page (Generic Desktop) */
	0x09, 0x06, /* Usage (Keyboard) */
	0xA1, 0x01, /* Collection (Application) */

	0x05, 0x07, /*   Usage Page (Keyboard/Keypad) */
	0x85, 0x01, /*   Report ID (1) */
	0x19, 0xE0, /*   Usage Minimum (Keyboard Left Control) */
	0x29, 0xE7, /*   Usage Maximum (Keyboard Right GUI) */
	0x15, 0x00, /*   Logical Minimum (0) */
	0x25, 0x01, /*   Logical Maximum (1) */
	0x75, 0x01, /*   Report Size (1) */
	0x95, 0x08, /*   Report Count (8) */
	0x81, 0x02, /*   Input (Data, Variable, Absolute) ; Modifier keys */

	0x95, 0x01, /*   Report Count (1) */
	0x75, 0x08, /*   Report Size (8) */
	0x81, 0x01, /*   Input (Constant) ; Reserved byte */

	0x95, 0x05, /*   Report Count (5) */
	0x75, 0x01, /*   Report Size (1) */
	0x05, 0x08, /*   Usage Page (LEDs) */
	0x85, 0x01, /*   Report ID (1) */
	0x19, 0x01, /*   Usage Minimum (Num Lock) */
	0x29, 0x05, /*   Usage Maximum (Kana) */
	0x91, 0x02, /*   Output (Data, Variable, Absolute) ; LED report */

	0x95, 0x01, /*   Report Count (1) */
	0x75, 0x03, /*   Report Size (3) */
	0x91, 0x03, /*   Output (Constant) ; LED report padding */

	0x95, 0x06, /*   Report Count (6) */
	0x75, 0x08, /*   Report Size (8) */
	0x15, 0x00, /*   Logical Minimum (0) */
	0x25, 0x65, /*   Logical Maximum (101) */
	0x05, 0x07, /*   Usage Page (Keyboard/Keypad) */
	0x19, 0x00, /*   Usage Minimum (Reserved (no event indicated)) */
	0x29, 0x65, /*   Usage Maximum (Keyboard Application) */
	0x81, 0x00, /*   Input (Data, Array) ; Key arrays (6 bytes) */

	0xC0, /* End Collection */

	/* Consumer Control Report */
	0x05, 0x0C, /* Usage Page (Consumer Devices) */
	0x09, 0x01, /* Usage (Consumer Control) */
	0xA1, 0x01, /* Collection (Application) */

	0x85, 0x02, /*   Report ID (2) */
	0x05, 0x0C, /*   Usage Page (Consumer Devices) */
	0x15, 0x00, /*   Logical Minimum (0) */
	0x25, 0x01, /*   Logical Maximum (1) */

	0x09, 0xE9, /*   Usage (Volume Up) */
	0x09, 0xEA, /*   Usage (Volume Down) */
	0x09, 0xE2, /*   Usage (Mute) */
	0x09, 0xCD, /*   Usage (Play/Pause) */
	0x19, 0xB5, /*   Usage Minimum (Scan Next Track) */
	0x29, 0xB8, /*   Usage Maximum (Scan Previous Track) */

	0x75, 0x01, /*   Report Size (1) */
	0x95, 0x08, /*   Report Count (8) */
	0x81, 0x02, /*   Input (Data, Variable, Absolute) ; Media keys */

	0xC0 /* End Collection */
};

static bool ble_connected;
static ble_connection_cb_t user_conn_callback;

static void connected(struct bt_conn *conn, uint8_t conn_err)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (conn_err) {
		LOG_ERR("Failed to connect to %s (%u)", addr, conn_err);
		return;
	}

	LOG_INF("Connected %s", addr);

	err = bt_hids_connected(&hids_obj, conn);
	if (err) {
		LOG_ERR("Failed to notify HIDS about connection (err %d)", err);
	}

	if (bt_conn_set_security(conn, BT_SECURITY_L2)) {
		LOG_ERR("Failed to set security");
	}

	ble_connected = true;

	if (user_conn_callback) {
		user_conn_callback(ble_connected);
	}
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_INF("Disconnected from %s (reason 0x%02x)", addr, reason);

	err = bt_hids_disconnected(&hids_obj, conn);
	if (err) {
		LOG_ERR("Failed to notify HIDS about disconnection (err %d)",
			err);
	}

	ble_connected = false;

	if (user_conn_callback) {
		user_conn_callback(ble_connected);
	}

	/* If disconnected due to authentication failure, clear all pairing info */
	if (reason == BT_HCI_ERR_AUTH_FAIL || reason == BT_HCI_ERR_PIN_OR_KEY_MISSING) {
		LOG_INF("Authentication related disconnect, clearing pairing info");
		bt_unpair(BT_ID_DEFAULT, BT_ADDR_LE_ANY);
	}

	/* Restart advertising */
	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	if (err) {
		LOG_ERR("Advertising failed to start (err %d)", err);
		return;
	}
	LOG_INF("Advertising successfully started");
}

static void security_changed(struct bt_conn *conn, bt_security_t level,
			     enum bt_security_err err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (!err) {
		LOG_INF("Security changed: %s level %u", addr, level);
	} else {
		LOG_ERR("Security failed: %s level %u err %d", addr, level, err);
		bt_unpair(BT_ID_DEFAULT, BT_ADDR_LE_ANY);
	}
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
	.security_changed = security_changed,
};

static int hids_service_init(void)
{
	struct bt_hids_init_param hids_init_param = {0};
	struct bt_hids_inp_rep *keyboard_rep;
	struct bt_hids_inp_rep *consumer_rep;

	hids_init_param.rep_map.data = report_map;
	hids_init_param.rep_map.size = sizeof(report_map);

	hids_init_param.info.bcd_hid = BASE_USB_HID_SPEC_VERSION;
	hids_init_param.info.b_country_code = 0x00;
	hids_init_param.info.flags = BT_HIDS_NORMALLY_CONNECTABLE;

	keyboard_rep = &hids_init_param.inp_rep_group_init.reports[INPUT_REP_KEYBOARD_IDX];
	keyboard_rep->size = INPUT_REP_KEYBOARD_LEN;
	keyboard_rep->id = INPUT_REP_KEYBOARD_ID;
	hids_init_param.inp_rep_group_init.cnt++;

	consumer_rep = &hids_init_param.inp_rep_group_init.reports[INPUT_REP_CONSUMER_IDX];
	consumer_rep->size = INPUT_REP_CONSUMER_LEN;
	consumer_rep->id = INPUT_REP_CONSUMER_ID;
	hids_init_param.inp_rep_group_init.cnt++;

	hids_init_param.is_kb = true;

	return bt_hids_init(&hids_obj, &hids_init_param);
}

static void bt_ready(int err)
{
	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return;
	}

	LOG_INF("Bluetooth initialized");

	if (IS_ENABLED(CONFIG_SETTINGS)) {
		settings_load();
	}

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	if (err) {
		LOG_ERR("Advertising failed to start (err %d)", err);
		return;
	}

	LOG_INF("Advertising successfully started");
}

int ble_hid_init(ble_connection_cb_t cb)
{
	int err;

	err = hids_service_init();

	if (err) {
		LOG_ERR("HID Service init failed (err %d)", err);
		return err;
	}

	err = bt_enable(bt_ready);

	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	user_conn_callback = cb;

	return err;
}

int ble_hid_send_key(ble_hid_key_t key)
{
	int err;
	uint8_t keyboard_report[INPUT_REP_KEYBOARD_LEN] = {0};
	uint8_t consumer_report[INPUT_REP_CONSUMER_LEN] = {0};
	bool is_consumer = false;

	if (!ble_connected) {
		return -ENOTCONN;
	}

	/* The keyboard report is sent as a single key pressed with modifier key bitmap cleared */
	switch (key) {
	case BLE_HID_KEY_ARROW_LEFT:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_ARROW_LEFT;
		break;
	case BLE_HID_KEY_ARROW_RIGHT:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_ARROW_RIGHT;
		break;
	case BLE_HID_KEY_F5:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_F5;
		break;
	case BLE_HID_KEY_ESC:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_ESC;
		break;
	case BLE_HID_KEY_MEDIA_PREV_TRACK:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_PREV_TRACK;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_NEXT_TRACK:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_NEXT_TRACK;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_MUTE:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_MUTE;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_PLAY_PAUSE:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_PLAY_PAUSE;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_VOLUME_UP:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_VOLUME_UP;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_VOLUME_DOWN:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_VOLUME_DOWN;
		is_consumer = true;
		break;
	default:
		return -EINVAL;
	}

	if (is_consumer) {
		err = bt_hids_inp_rep_send(&hids_obj, NULL,
					   INPUT_REP_CONSUMER_IDX,
					   consumer_report,
					   sizeof(consumer_report), NULL);
		if (err) {
			LOG_ERR("Failed to send consumer key (err %d)", err);
			return err;
		}

		memset(consumer_report, 0, sizeof(consumer_report));
		err = bt_hids_inp_rep_send(&hids_obj, NULL,
					   INPUT_REP_CONSUMER_IDX,
					   consumer_report,
					   sizeof(consumer_report), NULL);
	} else {
		err = bt_hids_inp_rep_send(&hids_obj, NULL,
					   INPUT_REP_KEYBOARD_IDX,
					   keyboard_report,
					   sizeof(keyboard_report), NULL);
		if (err) {
			LOG_ERR("Failed to send keyboard key (err %d)", err);
			return err;
		}

		memset(keyboard_report, 0, sizeof(keyboard_report));
		err = bt_hids_inp_rep_send(&hids_obj, NULL,
					   INPUT_REP_KEYBOARD_IDX,
					   keyboard_report,
					   sizeof(keyboard_report), NULL);
	}

	if (err) {
		LOG_ERR("Failed to send key release (err %d)", err);
		return err;
	}

	LOG_INF("BLE HID key sent successfully");

	return 0;
}
