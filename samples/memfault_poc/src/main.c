/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Memfault POC: report 5 bin counts (psi_bin_1 … psi_bin_5) as heartbeat
 * metrics. Values are set once per heartbeat using the Memfault metrics API.
 * Data is exposed via BLE MDS; a gateway forwards chunks to Memfault.
 */

#include <zephyr/kernel.h>
#include <zephyr/random/random.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/mds.h>
#include <memfault/metrics/metrics.h>

#define DEVICE_NAME     CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

#define PSI_BIN_COUNT  5
#define PSI_BIN_MAX    257   /* 0..256 inclusive */

static void security_changed(struct bt_conn *conn, bt_security_t level, enum bt_security_err err);

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
	(void)work;
	int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_2, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	if (err) {
		printk("Advertising start failed: %d\n", err);
		return;
	}
	printk("Advertising started\n");
}

static void advertising_start(void)
{
	k_work_submit(&adv_work);
}

static void connected(struct bt_conn *conn, uint8_t conn_err)
{
	if (conn_err) {
		printk("Connection failed: 0x%02x\n", conn_err);
		return;
	}
	mds_conn = conn;
	printk("Connected\n");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	printk("Disconnected: 0x%02x\n", reason);
	if (conn == mds_conn) {
		mds_conn = NULL;
	}
}

static void recycled_cb(void)
{
	advertising_start();
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
	.security_changed = security_changed,
	.recycled = recycled_cb,
};

static void security_changed(struct bt_conn *conn, bt_security_t level, enum bt_security_err err)
{
	if (err) {
		return;
	}
	if (level >= BT_SECURITY_L2 && !mds_conn) {
		mds_conn = conn;
	}
}

static void pairing_complete(struct bt_conn *conn, bool bonded)
{
	ARG_UNUSED(conn);
	ARG_UNUSED(bonded);
}

static void pairing_failed(struct bt_conn *conn, enum bt_security_err reason)
{
	ARG_UNUSED(conn);
	ARG_UNUSED(reason);
}

static void auth_cancel(struct bt_conn *conn)
{
	ARG_UNUSED(conn);
}

static struct bt_conn_auth_cb conn_auth_cb = {
	.cancel = auth_cancel,
};
static struct bt_conn_auth_info_cb conn_auth_info_cb = {
	.pairing_complete = pairing_complete,
	.pairing_failed = pairing_failed,
};

static bool mds_access_enable(struct bt_conn *conn)
{
	return (mds_conn != NULL && conn == mds_conn);
}

static const struct bt_mds_cb mds_cb = {
	.access_enable = mds_access_enable,
};

/* Called once per heartbeat; sets psi_bin_1..5 with current bin counts. */
void memfault_metrics_heartbeat_collect_data(void)
{
	uint32_t bin_counts[PSI_BIN_COUNT];
	printk("Collecting heartbeat data\n");

	for (int i = 0; i < PSI_BIN_COUNT; i++) {
		bin_counts[i] = (uint32_t)(sys_rand32_get() % PSI_BIN_MAX);
	}

	MEMFAULT_METRIC_SET_UNSIGNED(psi_bin_1, bin_counts[0]);
	MEMFAULT_METRIC_SET_UNSIGNED(psi_bin_2, bin_counts[1]);
	MEMFAULT_METRIC_SET_UNSIGNED(psi_bin_3, bin_counts[2]);
	MEMFAULT_METRIC_SET_UNSIGNED(psi_bin_4, bin_counts[3]);
	MEMFAULT_METRIC_SET_UNSIGNED(psi_bin_5, bin_counts[4]);
}

int main(void)
{
	int err;

	printk("Memfault POC (nRF54L20A) - BLE MDS, 5 psi bins in heartbeat\n");

	err = bt_mds_cb_register(&mds_cb);
	if (err) {
		printk("MDS callback registration failed: %d\n", err);
		return 0;
	}

	err = bt_enable(NULL);
	if (err) {
		printk("Bluetooth init failed: %d\n", err);
		return 0;
	}

	err = bt_conn_auth_cb_register(&conn_auth_cb);
	if (err) {
		printk("Auth cb register failed: %d\n", err);
	}
	err = bt_conn_auth_info_cb_register(&conn_auth_info_cb);
	if (err) {
		printk("Auth info cb register failed: %d\n", err);
	}

	if (IS_ENABLED(CONFIG_BT_SETTINGS)) {
		settings_load();
	}

	k_work_init(&adv_work, adv_work_handler);
	advertising_start();

	for (;;) {
		k_sleep(K_MSEC(1000));
	}
}
