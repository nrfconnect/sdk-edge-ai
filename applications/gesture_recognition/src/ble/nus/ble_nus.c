/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_nus.h"

#include <errno.h>
#include <string.h>
#include <stdio.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/logging/log.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/nus.h>

LOG_MODULE_REGISTER(ble_nus, LOG_LEVEL_INF);

static struct bt_conn *nus_conn;
static bool nus_send_enabled;
static ble_connection_cb_t user_conn_cb;

static const struct bt_data nus_ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static const struct bt_data nus_sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_VAL),
};

static void nus_send_enabled_cb(enum bt_nus_send_status status)
{
	nus_send_enabled = (status == BT_NUS_SEND_STATUS_ENABLED);
}

static void nus_connected(struct bt_conn *conn, uint8_t err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (err) {
		LOG_ERR("NUS connection failed to %s (%u)", addr, err);
		return;
	}

	if (!nus_conn) {
		nus_conn = bt_conn_ref(conn);
	}

	if (user_conn_cb) {
		user_conn_cb(true);
	}
	LOG_INF("NUS connected %s", addr);
}

static void nus_disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("NUS disconnected from %s (reason 0x%02x)", addr, reason);

	if (nus_conn == conn) {
		bt_conn_unref(nus_conn);
		nus_conn = NULL;
	}

	nus_send_enabled = false;

	if (user_conn_cb) {
		user_conn_cb(false);
	}

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad),
			     nus_sd, ARRAY_SIZE(nus_sd));
	if (err) {
		LOG_ERR("NUS Advertising failed to start (err %d)", err);
	}
}

static struct bt_nus_cb nus_cb = {
	.send_enabled = nus_send_enabled_cb,
};

static struct bt_conn_cb nus_conn_callbacks = {
	.connected = nus_connected,
	.disconnected = nus_disconnected,
};

int ble_nus_init(ble_connection_cb_t cb)
{
	int err;

	user_conn_cb = cb;

	err = bt_enable(NULL);
	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	LOG_INF("Bluetooth initialized");

	if (IS_ENABLED(CONFIG_SETTINGS)) {
		settings_load();
	}

	err = bt_nus_init(&nus_cb);
	if (err) {
		LOG_ERR("NUS init failed (err %d)", err);
		return err;
	}

	bt_conn_cb_register(&nus_conn_callbacks);

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad),
			     nus_sd, ARRAY_SIZE(nus_sd));
	if (err) {
		LOG_ERR("NUS advertising failed to start (err %d)", err);
		return err;
	}

	LOG_INF("NUS Advertising successfully started");
	return 0;
}

int ble_nus_send(const int16_t *input_data)
{
	char buffer[64];
	int len;
	uint32_t mtu;

	if (input_data == NULL) {
		return -EINVAL;
	}

	if (!nus_conn || !nus_send_enabled) {
		return -ENOTCONN;
	}

	len = snprintf(buffer, sizeof(buffer), "%d,%d,%d,%d,%d,%d\r\n",
		       input_data[0], input_data[1], input_data[2],
		       input_data[3], input_data[4], input_data[5]);
	if ((len <= 0) || (len >= (int)sizeof(buffer))) {
		return -EINVAL;
	}

	mtu = bt_nus_get_mtu(nus_conn);
	if ((uint32_t)len > mtu) {
		return -EMSGSIZE;
	}

	return bt_nus_send(nus_conn, (const uint8_t *)buffer, (uint16_t)len);
}
