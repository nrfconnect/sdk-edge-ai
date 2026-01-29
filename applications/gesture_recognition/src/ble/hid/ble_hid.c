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

LOG_MODULE_REGISTER(ble_hid, CONFIG_BT_HIDS_LOG_LEVEL);


#define KEY_ARROW_LEFT (0x50)
#define KEY_ARROW_RIGHT (0x4F)
#define KEY_F5 (0x3E)
#define KEY_ESC (0x29)

#define KEY_MEDIA_VOLUME_UP ( 1 << 0 )
#define KEY_MEDIA_VOLUME_DOWN ( 1 << 1 )
#define KEY_MEDIA_MUTE ( 1 << 2 )
#define KEY_MEDIA_PLAY_PAUSE ( 1 << 3 )
#define KEY_MEDIA_PREV_TRACK ( 1 << 4 )
#define KEY_MEDIA_NEXT_TRACK ( 1 << 5 )

/* Require encryption. */
#define SAMPLE_BT_PERM_READ BT_GATT_PERM_READ_ENCRYPT
#define SAMPLE_BT_PERM_WRITE BT_GATT_PERM_WRITE_ENCRYPT


enum
{
	HIDS_REMOTE_WAKE = BIT(0),
	HIDS_NORMALLY_CONNECTABLE = BIT(1),
};

struct hids_info
{
	uint16_t version; /* version number of base USB HID Specification */
	uint8_t code;     /* country HID Device hardware is localized for. */
	uint8_t flags;
} __packed;

struct hids_report
{
	uint8_t id;   /* report id */
	uint8_t type; /* report type */
} __packed;

enum
{
	HIDS_INPUT = 0x01,
	HIDS_OUTPUT = 0x02,
	HIDS_FEATURE = 0x03,
};

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID16_ALL,
				  BT_UUID_16_ENCODE(BT_UUID_HIDS_VAL),
				  BT_UUID_16_ENCODE(BT_UUID_BAS_VAL)),
};

static const struct bt_data sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static struct hids_info info = {
	.version = 0x0000,
	.code = 0x00,
	.flags = HIDS_NORMALLY_CONNECTABLE,
};

static struct hids_report input = {
	.id = 0x01,
	.type = HIDS_INPUT,
};

static struct hids_report input_consumer = {
	.id = 0x02,
	.type = HIDS_INPUT,
};

static bool ble_connected_ = false;
static ble_connection_cb_t user_conn_callback_ = NULL;

static bool ccc_enabled_ = false;
static uint8_t ctrl_point;
static uint8_t consumer_report;

static uint8_t report_map[] = {
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

static ssize_t read_info(struct bt_conn *conn,
			 const struct bt_gatt_attr *attr, void *buf,
			 uint16_t len, uint16_t offset)
{
	return bt_gatt_attr_read(conn, attr, buf, len, offset, attr->user_data,
				 sizeof(struct hids_info));
}

static ssize_t read_report_map(struct bt_conn *conn,
			       const struct bt_gatt_attr *attr, void *buf,
			       uint16_t len, uint16_t offset)
{
	return bt_gatt_attr_read(conn, attr, buf, len, offset, report_map,
				 sizeof(report_map));
}

static ssize_t read_report(struct bt_conn *conn,
			    const struct bt_gatt_attr *attr, void *buf,
			    uint16_t len, uint16_t offset)
{
	return bt_gatt_attr_read(conn, attr, buf, len, offset, attr->user_data,
				 sizeof(struct hids_report));
}

static void input_ccc_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
	LOG_INF("Input CCCD %s", value == BT_GATT_CCC_NOTIFY ? "enabled" : "disabled");
	LOG_INF("Input attribute handle: %d", attr->handle);

	ccc_enabled_ = (value == BT_GATT_CCC_NOTIFY);
}

static ssize_t read_input_report(struct bt_conn *conn,
				 const struct bt_gatt_attr *attr, void *buf,
				 uint16_t len, uint16_t offset)
{
	return bt_gatt_attr_read(conn, attr, buf, len, offset, NULL, 0);
}

static ssize_t write_ctrl_point(struct bt_conn *conn,
				const struct bt_gatt_attr *attr,
				const void *buf, uint16_t len, uint16_t offset,
				uint8_t flags)
{
	uint8_t *value = attr->user_data;

	LOG_INF("write_ctrl_point");

	if (offset + len > sizeof(ctrl_point)) {
		return BT_GATT_ERR(BT_ATT_ERR_INVALID_OFFSET);
	}

	memcpy(value + offset, buf, len);

	return len;
}

static ssize_t read_consumer_report(struct bt_conn *conn,
				    const struct bt_gatt_attr *attr,
				    void *buf, uint16_t len, uint16_t offset)
{
	return bt_gatt_attr_read(conn, attr, buf, len, offset, NULL, 0);
}

static void consumer_ccc_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
	LOG_INF("Consumer CCCD %s", value == BT_GATT_CCC_NOTIFY ? "enabled" : "disabled");
}

BT_GATT_SERVICE_DEFINE(hog_svc,
		       BT_GATT_PRIMARY_SERVICE(BT_UUID_HIDS),
		       BT_GATT_CHARACTERISTIC(BT_UUID_HIDS_INFO, BT_GATT_CHRC_READ,
					      BT_GATT_PERM_READ, read_info, NULL, &info),
		       BT_GATT_CHARACTERISTIC(BT_UUID_HIDS_REPORT_MAP, BT_GATT_CHRC_READ,
					      BT_GATT_PERM_READ, read_report_map, NULL, NULL),

		       BT_GATT_CHARACTERISTIC(BT_UUID_HIDS_REPORT,
					      BT_GATT_CHRC_READ | BT_GATT_CHRC_NOTIFY,
					      SAMPLE_BT_PERM_READ,
					      read_input_report, NULL, NULL),
		       BT_GATT_CCC(input_ccc_changed,
				   SAMPLE_BT_PERM_READ | SAMPLE_BT_PERM_WRITE),
		       BT_GATT_DESCRIPTOR(BT_UUID_HIDS_REPORT_REF, BT_GATT_PERM_READ,
					  read_report, NULL, &input),

		       BT_GATT_CHARACTERISTIC(BT_UUID_HIDS_REPORT,
					      BT_GATT_CHRC_READ | BT_GATT_CHRC_NOTIFY,
					      SAMPLE_BT_PERM_READ,
					      read_consumer_report, NULL, &consumer_report),
		       BT_GATT_CCC(consumer_ccc_changed,
				   SAMPLE_BT_PERM_READ | SAMPLE_BT_PERM_WRITE),
		       BT_GATT_DESCRIPTOR(BT_UUID_HIDS_REPORT_REF, BT_GATT_PERM_READ,
					  read_report, NULL, &input_consumer),

		       BT_GATT_CHARACTERISTIC(BT_UUID_HIDS_CTRL_POINT,
					      BT_GATT_CHRC_WRITE_WITHOUT_RESP,
					      BT_GATT_PERM_WRITE,
					      NULL, write_ctrl_point, &ctrl_point), );

static void connected(struct bt_conn *conn, uint8_t err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (err) {
		LOG_ERR("Failed to connect to %s (%u)", addr, err);
		return;
	}

	LOG_INF("Connected %s", addr);

	if (bt_conn_set_security(conn, BT_SECURITY_L2)) {
		LOG_ERR("Failed to set security");
	}

	ble_connected_ = true;

	if (user_conn_callback_) {
		user_conn_callback_(ble_connected_);
	}
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_INF("Disconnected from %s (reason 0x%02x)", addr, reason);

	ble_connected_ = false;
	ccc_enabled_ = false;

	if (user_conn_callback_) {
		user_conn_callback_(ble_connected_);
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

static void auth_passkey_display(struct bt_conn *conn, unsigned int passkey)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_INF("Passkey for %s: %06u", addr, passkey);
}

static void auth_cancel(struct bt_conn *conn)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_INF("Pairing cancelled: %s", addr);
}

static struct bt_conn_auth_cb auth_cb_display = {
	.passkey_display = auth_passkey_display,
	.passkey_entry = NULL,
	.cancel = auth_cancel,
};

int ble_hid_init(ble_connection_cb_t cb)
{
	int err;

	err = bt_enable(bt_ready);

	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	user_conn_callback_ = cb;

	if (IS_ENABLED(CONFIG_SAMPLE_BT_USE_AUTHENTICATION)) {
		bt_conn_auth_cb_register(&auth_cb_display);
	}

	return err;
}

int ble_hid_send_key(ble_hid_key_t key)
{
	int res = -1;
	uint8_t report[8] = {0};
	size_t report_len = sizeof(report);
	int attr_index = 5;

	if (!ble_connected_ || !ccc_enabled_) {
		return -1;
	}

	switch (key) {
	case BLE_HID_KEY_ARROW_LEFT:
		report[2] = KEY_ARROW_LEFT;
		break;
	case BLE_HID_KEY_ARROW_RIGHT:
		report[2] = KEY_ARROW_RIGHT;
		break;
	case BLE_HID_KEY_F5:
		report[2] = KEY_F5;
		break;
	case BLE_HID_KEY_ESC:
		report[2] = KEY_ESC;
		break;
	case BLE_HID_KEY_MEDIA_PREV_TRACK:
		report[0] = KEY_MEDIA_PREV_TRACK;
		report_len = 2;
		attr_index = 10;
		break;
	case BLE_HID_KEY_MEDIA_NEXT_TRACK:
		report[0] = KEY_MEDIA_NEXT_TRACK;
		report_len = 2;
		attr_index = 10;
		break;
	case BLE_HID_KEY_MEDIA_MUTE:
		report[0] = KEY_MEDIA_MUTE;
		report_len = 2;
		attr_index = 10;
		break;
	case BLE_HID_KEY_MEDIA_PLAY_PAUSE:
		report[0] = KEY_MEDIA_PLAY_PAUSE;
		report_len = 2;
		attr_index = 10;
		break;
	case BLE_HID_KEY_MEDIA_VOLUME_UP:
		report[0] = KEY_MEDIA_VOLUME_UP;
		report_len = 2;
		attr_index = 10;
		break;
	case BLE_HID_KEY_MEDIA_VOLUME_DOWN:
		report[0] = KEY_MEDIA_VOLUME_DOWN;
		report_len = 2;
		attr_index = 10;
		break;
	default:
		return res;
	}

	res = bt_gatt_notify(NULL, &hog_svc.attrs[attr_index], report, report_len);

	if (res) {
		LOG_ERR("Failed to send key, error = %d", res);
		return res;
	} else {
		LOG_INF("BLE HID Key %d sent successfully", report[0]);
	}

	/* reset report */
	memset(report, 0, sizeof(report));

	res = bt_gatt_notify(NULL, &hog_svc.attrs[attr_index], report, report_len);
	return res;
}
