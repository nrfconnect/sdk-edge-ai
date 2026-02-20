/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Memfault POC: every 60 seconds record 5 random numbers (0–256) as a Custom
 * Data Recording (CDR) and expose them via BLE MDS. A gateway (nRF Memfault app,
 * Web Bluetooth at mflt.io/mds, or nRF Python script) forwards chunks to Memfault.
 */

#include <zephyr/kernel.h>
#include <zephyr/random/random.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gap.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/mds.h>

#include <memfault/core/custom_data_recording.h>

#define DEVICE_NAME     CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

#define RANDOM_COUNT  5
#define RANDOM_MAX   257   /* 0..256 inclusive */
#define CDR_INTERVAL_SEC 60

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
static struct k_work_delayable cdr_work;

/* CDR: 5 bytes, one random value 0..256 per byte */
static uint8_t cdr_payload[RANDOM_COUNT];
static bool cdr_ready_to_send;
static size_t cdr_read_offset;

static const char *cdr_mimetypes[] = { MEMFAULT_CDR_BINARY };
static sMemfaultCdrMetadata cdr_metadata = {
	.start_time = { .type = kMemfaultCurrentTimeType_Unknown },
	.mimetypes = cdr_mimetypes,
	.num_mimetypes = ARRAY_SIZE(cdr_mimetypes),
	.data_size_bytes = RANDOM_COUNT,
	.duration_ms = 0,
	.collection_reason = "memfault_poc_random_sample",
};

static bool cdr_has_cb(sMemfaultCdrMetadata *metadata)
{
	*metadata = cdr_metadata;
	return cdr_ready_to_send;
}

static bool cdr_read_cb(uint32_t offset, void *data, size_t data_len)
{
	if (offset != cdr_read_offset) {
		return false;
	}
	size_t copy_len = (data_len < (RANDOM_COUNT - offset)) ? data_len : (RANDOM_COUNT - offset);
	memcpy(data, &cdr_payload[offset], copy_len);
	cdr_read_offset += copy_len;
	return true;
}

static void cdr_mark_read_cb(void)
{
	cdr_ready_to_send = false;
	cdr_read_offset = 0;
}

static const sMemfaultCdrSourceImpl cdr_source = {
	.has_cdr_cb = cdr_has_cb,
	.read_data_cb = cdr_read_cb,
	.mark_cdr_read_cb = cdr_mark_read_cb,
};

static void cdr_work_handler(struct k_work *work)
{
	(void)work;
	for (int i = 0; i < RANDOM_COUNT; i++) {
		cdr_payload[i] = (uint8_t)(sys_rand32_get() % RANDOM_MAX);
	}
	cdr_read_offset = 0;
	cdr_ready_to_send = true;
	k_work_reschedule(&cdr_work, K_SECONDS(CDR_INTERVAL_SEC));
}

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
	/* Restart advertising only in recycled_cb() when the connection is fully torn down */
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

int main(void)
{
	int err;

	printk("Memfault POC (nRF54L20A) - BLE MDS, 5 randoms every %ds\n", CDR_INTERVAL_SEC);

	if (!memfault_cdr_register_source(&cdr_source)) {
		printk("CDR registration failed\n");
		return 0;
	}

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
	k_work_init_delayable(&cdr_work, cdr_work_handler);

	advertising_start();
	/* First CDR in 60 s, then every 60 s */
	k_work_reschedule(&cdr_work, K_SECONDS(CDR_INTERVAL_SEC));

	for (;;) {
		k_sleep(K_MSEC(1000));
	}
}
