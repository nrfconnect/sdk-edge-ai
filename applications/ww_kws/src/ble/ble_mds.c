/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/mds.h>

LOG_MODULE_REGISTER(ble_mds);

#define DEVICE_NAME	CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_MDS_VAL),
};

static const struct bt_data sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};

static struct bt_conn *mds_conn;
static struct k_work adv_work;

static void adv_work_handler(struct k_work *work)
{
	int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_2, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));

	if (err) {
		LOG_ERR("Advertising failed to start (err %d)", err);
		return;
	}

	LOG_INF("Advertising successfully started");
}

static void advertising_start(void)
{
	k_work_submit(&adv_work);
}

static void connected(struct bt_conn *conn, uint8_t conn_err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	if (conn_err) {
		LOG_ERR("Connection failed, err 0x%02x %s", conn_err, bt_hci_err_to_str(conn_err));
		return;
	}

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Connected %s", addr);
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	LOG_INF("Disconnected, reason 0x%02x %s", reason, bt_hci_err_to_str(reason));

	if (conn == mds_conn) {
		mds_conn = NULL;
	}
}

static void security_changed(struct bt_conn *conn, bt_security_t level, enum bt_security_err err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (!err) {
		LOG_INF("Security changed: %s level %u", addr, level);
	} else {
		LOG_ERR("Security failed: %s level %u err %d %s", addr, level, err,
			bt_security_err_to_str(err));
	}

	if (level >= BT_SECURITY_L2 && !mds_conn) {
		mds_conn = conn;
	}
}

static void recycled_cb(void)
{
	LOG_DBG("Connection object available, restarting advertising");
	advertising_start();
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
	.security_changed = security_changed,
	.recycled = recycled_cb,
};

static bool mds_access_enable(struct bt_conn *conn)
{
	return mds_conn && (conn == mds_conn);
}

static const struct bt_mds_cb mds_cb = {
	.access_enable = mds_access_enable,
};

int init_app_ble(void)
{
	int err;

	err = bt_mds_cb_register(&mds_cb);
	if (err) {
		LOG_ERR("MDS callback registration failed (err %d)", err);
		return err;
	}

	err = bt_enable(NULL);
	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	LOG_INF("Bluetooth initialized");

	if (IS_ENABLED(CONFIG_SETTINGS)) {
		err = settings_load();
		if (err) {
			LOG_ERR("Failed to load settings (err %d)", err);
			return err;
		}
	}

	k_work_init(&adv_work, adv_work_handler);
	advertising_start();

	return 0;
}
