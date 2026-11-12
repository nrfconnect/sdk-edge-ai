/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_nus.h"

#include <errno.h>
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
static ble_nus_conn_cb_t nus_conn_cb;

static const struct bt_data nus_ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static const struct bt_data nus_sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_VAL),
};

static int nus_adv_start(void)
{
	return bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad), nus_sd,
			       ARRAY_SIZE(nus_sd));
}

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

	LOG_INF("NUS connected %s", addr);

	if (nus_conn_cb) {
		nus_conn_cb(true);
	}
}

static void nus_disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("NUS disconnected from %s (reason 0x%02x)", addr, reason);

	if (nus_conn == conn) {
		bt_conn_unref(nus_conn);
		nus_conn = NULL;
	}

	nus_send_enabled = false;

	if (nus_conn_cb) {
		nus_conn_cb(false);
	}
}

/* Restart advertising once the connection object is released */
static void nus_recycled(void)
{
	int err = nus_adv_start();

	if (err) {
		LOG_ERR("NUS advertising failed to start (err %d)", err);
	}
}

static struct bt_nus_cb nus_cb = {
	.send_enabled = nus_send_enabled_cb,
};

BT_CONN_CB_DEFINE(nus_conn_callbacks) = {
	.connected = nus_connected,
	.disconnected = nus_disconnected,
	.recycled = nus_recycled,
};

int ble_nus_init(ble_nus_conn_cb_t conn_cb)
{
	int err;

	nus_conn_cb = conn_cb;

	err = bt_enable(NULL);
	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	LOG_INF("Bluetooth initialized");

	/* With CONFIG_BT_SETTINGS, Bluetooth is usable only after the settings are loaded */
	if (IS_ENABLED(CONFIG_SETTINGS)) {
		err = settings_load();
		if (err) {
			LOG_ERR("Failed to load settings (err %d)", err);
			return err;
		}
	}

	err = bt_nus_init(&nus_cb);
	if (err) {
		LOG_ERR("NUS init failed (err %d)", err);
		return err;
	}

	err = nus_adv_start();
	if (err) {
		LOG_ERR("NUS advertising failed to start (err %d)", err);
		return err;
	}

	LOG_INF("NUS Advertising successfully started");
	return 0;
}

int ble_nus_send_result(uint16_t class_id, uint8_t probability_pct)
{
	char buffer[16];
	int len;

	if (!nus_conn || !nus_send_enabled) {
		return -ENOTCONN;
	}

	len = snprintf(buffer, sizeof(buffer), "%u,%u\r\n", class_id, probability_pct);
	if ((len <= 0) || (len >= (int)sizeof(buffer))) {
		return -EINVAL;
	}

	if ((uint32_t)len > bt_nus_get_mtu(nus_conn)) {
		return -EMSGSIZE;
	}

	return bt_nus_send(nus_conn, (const uint8_t *)buffer, (uint16_t)len);
}
