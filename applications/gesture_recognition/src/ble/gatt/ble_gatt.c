/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_gatt.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/settings/settings.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(ble_gatt, LOG_LEVEL_INF);

static ssize_t on_data_received(struct bt_conn *conn,
				 const struct bt_gatt_attr *attr,
				 const void *buf,
				 uint16_t len,
				 uint16_t offset,
				 uint8_t flags);

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID16_ALL, BT_UUID_16_ENCODE(BT_UUID_BAS_VAL)),
};

static const struct bt_data sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static bool ble_connected;
static ble_connection_cb_t user_conn_callback;
static ble_data_received_cb_t user_data_received_callback;

static bool ccc_enabled;

static struct bt_conn *current_conn;

static void on_cccd_changed(const struct bt_gatt_attr *attr, uint16_t value)
{
	LOG_INF("Input CCCD %s", value == BT_GATT_CCC_NOTIFY ? "enabled" : "disabled");
	LOG_INF("Input attribute handle: %d", attr->handle);

	ccc_enabled = (value == BT_GATT_CCC_NOTIFY);
}

/* Declaration of custom GATT service and characteristics UUIDs */
#define NEUTON_SERVICE_UUID \
	BT_UUID_128_ENCODE(0xa5d4f351, 0x9d11, 0x419f, 0x9f1b, 0x3dcdf0a15f4d)

#define NEUTON_OUT_CHARACTERISTIC_UUID \
	BT_UUID_128_ENCODE(0x516a51c4, 0xb1e1, 0x47fa, 0x8327, 0x8acaeb3399eb)

#define NEUTON_IN_CHARACTERISTIC_UUID \
	BT_UUID_128_ENCODE(0x516a51c4, 0xb1e1, 0x47fa, 0x8327, 0x8acaeb3399ec)

#define BT_UUID_NEUTON_SERVICE		BT_UUID_DECLARE_128(NEUTON_SERVICE_UUID)
#define BT_UUID_NEUTON_CHAR_OUT		BT_UUID_DECLARE_128(NEUTON_OUT_CHARACTERISTIC_UUID)
#define BT_UUID_NEUTON_CHAR_IN		BT_UUID_DECLARE_128(NEUTON_IN_CHARACTERISTIC_UUID)

/* Neuton GATT Service Declaration and Registration */
BT_GATT_SERVICE_DEFINE(neuton_gatt,
	BT_GATT_PRIMARY_SERVICE(BT_UUID_NEUTON_SERVICE),
	BT_GATT_CHARACTERISTIC(BT_UUID_NEUTON_CHAR_OUT,
		BT_GATT_CHRC_NOTIFY,
		BT_GATT_PERM_NONE,
		NULL, NULL, NULL),
	BT_GATT_CCC(on_cccd_changed, BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
	BT_GATT_CHARACTERISTIC(BT_UUID_NEUTON_CHAR_IN,
		BT_GATT_CHRC_WRITE_WITHOUT_RESP,
		BT_GATT_PERM_WRITE,
		NULL, on_data_received, NULL)
);

static void connected(struct bt_conn *conn, uint8_t err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (err) {
		LOG_ERR("Failed to connect to %s (%u)", addr, err);
		return;
	}

	LOG_INF("Connected %s", addr);
	ble_connected = true;
	if (!current_conn) {
		current_conn = bt_conn_ref(conn);
	}
	if (user_conn_callback) {
		user_conn_callback(ble_connected);
	}
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Disconnected from %s (reason 0x%02x)", addr, reason);
	ble_connected = false;
	ccc_enabled = false;
	if (current_conn) {
		bt_conn_unref(current_conn);
		current_conn = NULL;
	}
	if (user_conn_callback) {
		user_conn_callback(ble_connected);
	}
	if (reason == BT_HCI_ERR_AUTH_FAIL || reason == BT_HCI_ERR_PIN_OR_KEY_MISSING) {
		LOG_INF("Authentication related disconnect, clearing pairing info");
		bt_unpair(BT_ID_DEFAULT, BT_ADDR_LE_ANY);
	}
	int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));

	if (err) {
		LOG_ERR("Advertising failed to start (err %d)", err);
		return;
	}
	LOG_INF("Advertising successfully started");
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
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

static ssize_t on_data_received(struct bt_conn *conn,
				const struct bt_gatt_attr *attr,
				const void *buf,
				uint16_t len,
				uint16_t offset,
				uint8_t flags)
{
	if (user_data_received_callback) {
		user_data_received_callback((const char *)buf, len);
	} else {
		LOG_ERR("Data received but no callback registered");
	}
	return len; /* Indicate all data was consumed */
}

static int read_conn_rssi(struct bt_conn *conn, int8_t *out_rssi)
{
	if (!conn || !out_rssi) {
		LOG_ERR("Invalid connection or RSSI pointer");
		return -EINVAL;
	}

	uint16_t handle;
	int err = bt_conn_get_info(conn, &(struct bt_conn_info){0});

	if (err) {
		LOG_ERR("Failed to get conn info (err %d)", err);
		return err;
	}

	/* Get the connection handle used by the controller */
	err = bt_hci_get_conn_handle(conn, &handle);
	if (err) {
		LOG_ERR("Failed to get HCI handle (err %d)", err);
		return err;
	}

	struct bt_hci_cp_read_rssi cp = {
		.handle = sys_cpu_to_le16(handle),
	};

	struct net_buf *rsp_buf;
	int ret;

	struct net_buf *buf = bt_hci_cmd_alloc(K_FOREVER);

	if (!buf) {
		LOG_ERR("No HCI buffer available");
		return -EINVAL;
	}

	net_buf_add_mem(buf, &cp, sizeof(cp));

	ret = bt_hci_cmd_send_sync(BT_HCI_OP_READ_RSSI, buf, &rsp_buf);

	if (ret) {
		LOG_ERR("HCI RSSI request failed (err %d)", ret);
		net_buf_unref(buf);
		return ret;
	}

	struct bt_hci_rp_read_rssi *rp = (struct bt_hci_rp_read_rssi *)rsp_buf->data;

	if (rp->status == 0) {
		*out_rssi = rp->rssi; /* value is already signed dBm */
	} else {
		LOG_ERR("HCI RSSI status error: 0x%02x", rp->status);
	}

	return rp->status;
}

int ble_gatt_init(ble_connection_cb_t connection_cb, ble_data_received_cb_t data_received_cb)
{
	int err;

	err = bt_enable(bt_ready);

	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}
	user_conn_callback = connection_cb;
	user_data_received_callback = data_received_cb;
	return err;
}

int ble_gatt_send_raw_data(const uint8_t *data, size_t len)
{
	int res;

	if (!ble_connected || !ccc_enabled) {
		return -ENOTCONN;
	}

	if (data == NULL || len == 0) {
		LOG_ERR("Invalid data or length");
		return -EINVAL;
	}

	const struct bt_gatt_attr *attr = &neuton_gatt.attrs[2];

	res = bt_gatt_notify(NULL, attr, data, len);

	return res;
}

int ble_gatt_start_advertising(void)
{
	if (ble_connected) {
		LOG_ERR("Cannot start advertising while connected");
		return -EBUSY;
	}

	int res = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));

	return res;
}

int ble_gatt_get_rssi(int8_t *out_rssi)
{
	int res;

	if (!current_conn) {

		LOG_ERR("No current connection");
		return -EINVAL;
	}

	int8_t rssi = 0;

	res = read_conn_rssi(current_conn, &rssi);

	if (res == 0) {
		if (out_rssi) {
			*out_rssi = rssi;
		}
	} else {
		LOG_ERR("Failed to read RSSI (err %d)", res);
	}
	return res;
}
