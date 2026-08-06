/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_nus.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/logging/log.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/nus.h>

#include "../ble_common.h"
#include "../../activity_led.h"

LOG_MODULE_REGISTER(ble_nus, LOG_LEVEL_INF);

static struct bt_conn *nus_conn;
static bool nus_send_enabled;

static const struct bt_data nus_ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static const struct bt_data nus_sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_VAL),
};

static int nus_send_string(const char *msg)
{
	size_t len;
	uint32_t mtu;
	int err;

	if (!nus_conn || !nus_send_enabled || (msg == NULL)) {
		return -ENOTCONN;
	}

	len = strlen(msg);
	mtu = bt_nus_get_mtu(nus_conn);
	if (len > mtu) {
		return -EMSGSIZE;
	}

	err = bt_nus_send(nus_conn, (const uint8_t *)msg, (uint16_t)len);
	if (err == 0) {
		activity_led_nus_tx_pulse();
	}

	return err;
}

static void nus_send_enabled_cb(enum bt_nus_send_status status)
{
	nus_send_enabled = (status == BT_NUS_SEND_STATUS_ENABLED);

	if (nus_send_enabled) {
		LOG_INF("NUS TX notifications enabled");
		(void)nus_send_string("CONNECTED\r\n");
	}
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

	ble_common_set_connected(true);
	activity_led_set_ble_connected(true);
	LOG_INF("NUS connected %s", addr);
}

static void nus_disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int ret;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("NUS disconnected from %s (reason 0x%02x)", addr, reason);

	if (nus_conn == conn) {
		bt_conn_unref(nus_conn);
		nus_conn = NULL;
	}

	nus_send_enabled = false;
	ble_common_set_connected(false);
	activity_led_set_ble_connected(false);

	ret = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad), nus_sd,
			      ARRAY_SIZE(nus_sd));
	if (ret) {
		LOG_ERR("NUS advertising failed to start (err %d)", ret);
	}
}

static struct bt_nus_cb nus_cb = {
	.send_enabled = nus_send_enabled_cb,
};

static struct bt_conn_cb nus_conn_callbacks = {
	.connected = nus_connected,
	.disconnected = nus_disconnected,
};

int ble_nus_init(void)
{
	int err;

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

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad), nus_sd,
			      ARRAY_SIZE(nus_sd));
	if (err) {
		LOG_ERR("NUS advertising failed to start (err %d)", err);
		return err;
	}

	LOG_INF("NUS advertising successfully started");
	return 0;
}

int ble_nus_send_message(const char *message)
{
	char buffer[48];
	int len;

	if (message == NULL) {
		return -EINVAL;
	}

	len = snprintf(buffer, sizeof(buffer), "%s\r\n", message);
	if ((len <= 0) || (len >= (int)sizeof(buffer))) {
		return -EINVAL;
	}

	return nus_send_string(buffer);
}

int ble_nus_send_classification(const char *prefix, const char *class_name, int probability_pct,
				float accel_x_g, float accel_y_g, float accel_z_g)
{
	char buffer[72];
	int len;

	if (class_name == NULL) {
		return -EINVAL;
	}

	len = snprintf(buffer, sizeof(buffer), "%s%s,%d,%.3f,%.3f,%.3f\r\n",
		       (prefix != NULL) ? prefix : "", class_name, probability_pct,
		       (double)accel_x_g, (double)accel_y_g, (double)accel_z_g);
	if ((len <= 0) || (len >= (int)sizeof(buffer))) {
		return -EINVAL;
	}

	return nus_send_string(buffer);
}
