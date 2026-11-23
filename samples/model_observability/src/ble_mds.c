/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_mds.h"

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/mds.h>

LOG_MODULE_REGISTER(ble_mds, LOG_LEVEL_INF);

static const struct bt_data mds_ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_MDS_VAL),
};

static const struct bt_data mds_sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

/* Connection allowed to access MDS, set once the link is encrypted (security >= L2) */
static struct bt_conn *mds_conn;
static ble_mds_conn_cb_t mds_conn_cb;
static struct k_work adv_work;

static void adv_work_handler(struct k_work *work)
{
	int err;

	ARG_UNUSED(work);

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_2, mds_ad, ARRAY_SIZE(mds_ad), mds_sd,
			      ARRAY_SIZE(mds_sd));
	if (err) {
		LOG_ERR("MDS advertising failed to start (err %d)", err);
		return;
	}

	LOG_INF("MDS advertising successfully started");
}

static void mds_connected(struct bt_conn *conn, uint8_t conn_err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	if (conn_err) {
		LOG_ERR("Connection failed, err 0x%02x %s", conn_err, bt_hci_err_to_str(conn_err));
		return;
	}

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Connected %s", addr);

	if (mds_conn_cb) {
		mds_conn_cb(true);
	}
}

static void mds_disconnected(struct bt_conn *conn, uint8_t reason)
{
	LOG_INF("Disconnected, reason 0x%02x %s", reason, bt_hci_err_to_str(reason));

	if (conn == mds_conn) {
		mds_conn = NULL;
	}

	if (mds_conn_cb) {
		mds_conn_cb(false);
	}
}

static void mds_security_changed(struct bt_conn *conn, bt_security_t level,
				 enum bt_security_err err)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (!err) {
		LOG_INF("Security changed: %s level %u", addr, level);
	} else {
		LOG_ERR("Security failed: %s level %u err %d %s", addr, level, err,
			bt_security_err_to_str(err));
	}

	if ((level >= BT_SECURITY_L2) && !mds_conn) {
		mds_conn = conn;
	}
}

/* Restart advertising once the connection object is released */
static void mds_recycled(void)
{
	k_work_submit(&adv_work);
}

BT_CONN_CB_DEFINE(mds_conn_callbacks) = {
	.connected = mds_connected,
	.disconnected = mds_disconnected,
	.security_changed = mds_security_changed,
	.recycled = mds_recycled,
};

static bool mds_access_enable(struct bt_conn *conn)
{
	return mds_conn && (conn == mds_conn);
}

static const struct bt_mds_cb mds_cb = {
	.access_enable = mds_access_enable,
};

int ble_mds_init(ble_mds_conn_cb_t conn_cb)
{
	int err;

	mds_conn_cb = conn_cb;

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
	k_work_submit(&adv_work);

	return 0;
}
