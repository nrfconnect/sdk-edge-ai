/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_services.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/logging/log.h>
#include <zephyr/settings/settings.h>
#include <zephyr/sys/atomic.h>

#include <bluetooth/services/nus.h>

#if IS_ENABLED(CONFIG_BT_MDS)
#include <bluetooth/services/mds.h>
#endif

#include "ble_common.h"
#include "../activity_led.h"

LOG_MODULE_REGISTER(ble_services, LOG_LEVEL_INF);

static bool nus_send_enabled;
static atomic_t connection_count;

static const struct bt_data nus_ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
#if IS_ENABLED(CONFIG_BT_MDS)
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_MDS_VAL),
#else
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
#endif
};

static const struct bt_data nus_sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_VAL),
#if IS_ENABLED(CONFIG_BT_MDS)
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
#endif
};

static int advertising_start(void)
{
	int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, nus_ad, ARRAY_SIZE(nus_ad), nus_sd,
				  ARRAY_SIZE(nus_sd));

	if (err == -EALREADY) {
		return 0;
	}

	return err;
}

static int nus_send_string(const char *msg)
{
	int err;

	if (!nus_send_enabled || (msg == NULL)) {
		return -ENOTCONN;
	}

	err = bt_nus_send(NULL, (const uint8_t *)msg, (uint16_t)strlen(msg));
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

static void connected(struct bt_conn *conn, uint8_t err)
{
	char addr[BT_ADDR_LE_STR_LEN];
	atomic_val_t conn_count;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (err) {
		LOG_ERR("Bluetooth connection failed to %s (%u)", addr, err);
		return;
	}

	conn_count = atomic_inc(&connection_count) + 1;
	ble_common_set_connected(true);
	activity_led_set_ble_connected(true);
	LOG_INF("Bluetooth connected %s (%ld/%d)", addr, conn_count, CONFIG_BT_MAX_CONN);

	if (conn_count < CONFIG_BT_MAX_CONN) {
		int adv_err = advertising_start();

		if (adv_err) {
			LOG_ERR("Bluetooth advertising failed to restart (err %d)", adv_err);
		}
	}
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	atomic_val_t conn_count;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Bluetooth disconnected from %s (reason 0x%02x)", addr, reason);

	conn_count = atomic_dec(&connection_count) - 1;
	ble_common_set_connected(conn_count > 0);
	activity_led_set_ble_connected(conn_count > 0);

	int adv_err = advertising_start();

	if (adv_err) {
		LOG_ERR("Bluetooth advertising failed to start (err %d)", adv_err);
	}
}

static struct bt_nus_cb nus_cb = {
	.send_enabled = nus_send_enabled_cb,
};

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
};

#if IS_ENABLED(CONFIG_BT_MDS)
static bool mds_access_enable(struct bt_conn *conn)
{
	return bt_conn_get_security(conn) >= BT_SECURITY_L2;
}

static const struct bt_mds_cb mds_cb = {
	.access_enable = mds_access_enable,
};
#endif

int ble_nus_init(void)
{
	int err;

#if IS_ENABLED(CONFIG_BT_MDS)
	err = bt_mds_cb_register(&mds_cb);
	if (err) {
		LOG_ERR("MDS callback registration failed (err %d)", err);
		return err;
	}
#endif

	err = bt_enable(NULL);
	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	LOG_INF("Bluetooth initialized");

	if (IS_ENABLED(CONFIG_SETTINGS)) {
		int ret = settings_load();

		if (ret) {
			LOG_WRN("Failed to load settings (err %d)", ret);
		}
	}

	err = bt_nus_init(&nus_cb);
	if (err) {
		LOG_ERR("NUS init failed (err %d)", err);
		return err;
	}

	err = advertising_start();
	if (err) {
		LOG_ERR("Bluetooth advertising failed to start (err %d)", err);
		return err;
	}

	LOG_INF("Bluetooth advertising successfully started");
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

int ble_nus_unpair(void)
{
	return bt_unpair(BT_ID_DEFAULT, NULL);
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
