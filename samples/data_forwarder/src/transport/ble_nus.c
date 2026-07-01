/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "transport.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/logging/log.h>

#include <bluetooth/services/nus.h>

LOG_MODULE_REGISTER(transport, CONFIG_LOG_DEFAULT_LEVEL);

#define DEVICE_NAME	CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

static struct bt_conn *current_conn;

static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};

static const struct bt_data sd[] = {
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_NUS_VAL),
};

static size_t nus_mtu;

static void nus_recv_cb(struct bt_conn *conn, const uint8_t *const data, uint16_t len)
{
	ARG_UNUSED(conn);
	ARG_UNUSED(data);
	ARG_UNUSED(len);
}

static struct bt_nus_cb nus_cb = {
	.received = nus_recv_cb,
};

static void bt_nus_exchange_cb(struct bt_conn *conn, uint8_t err,
			       struct bt_gatt_exchange_params *params)
{
	if (err) {
		LOG_WRN("MTU exchange failed (err %u)", err);
	} else {
		nus_mtu = bt_nus_get_mtu(conn);
		LOG_INF("New NUS MTU: %d bytes", nus_mtu);
	}
}

static void connected(struct bt_conn *conn, uint8_t err)
{
	if (err) {
		LOG_WRN("BLE connect failed (err %u)", err);
		return;
	}

	if (current_conn == NULL) {
		current_conn = bt_conn_ref(conn);
	}

	static struct bt_gatt_exchange_params params = {.func = bt_nus_exchange_cb};

	bt_gatt_exchange_mtu(conn, &params);

	LOG_INF("BLE connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	ARG_UNUSED(reason);

	if (current_conn == conn) {
		bt_conn_unref(current_conn);
		current_conn = NULL;
	}

	nus_mtu = 0;

	LOG_INF("BLE disconnected");
}

static void recycled(void)
{
	int err;

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	if (err) {
		LOG_ERR("Advertising start failed (err %d)", err);
	}
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
	.recycled = recycled,
};

static int ble_send_cb(const uint8_t *buf, size_t len, void *ctx)
{
	size_t off = 0U;
	int err;

	if ((buf == NULL) || (len == 0U) || (nus_mtu == 0U)) {
		return -EINVAL;
	}

	if (current_conn == NULL) {
		return -ENOTCONN;
	}

	while (off < len) {
		const size_t chunk = MIN(nus_mtu, len - off);

		err = bt_nus_send(current_conn, &buf[off], chunk);
		if (err) {
			return err;
		}
		off += chunk;
	}

	return 0;
}

int transport_init(struct proto_transport *out_transport)
{
	int err;

	if (out_transport == NULL) {
		return -EINVAL;
	}

	err = bt_enable(NULL);
	if (err) {
		LOG_ERR("bt_enable failed (err %d)", err);
		return err;
	}

	err = bt_nus_init(&nus_cb);
	if (err) {
		LOG_ERR("bt_nus_init failed (err %d)", err);
		return err;
	}

	err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_2, ad, ARRAY_SIZE(ad), sd, ARRAY_SIZE(sd));
	if (err) {
		LOG_ERR("Advertising start failed (err %d)", err);
		return err;
	}

	out_transport->send = ble_send_cb;
	out_transport->ctx = NULL;
	out_transport->has_message_boundaries = false;

	LOG_INF("BLE NUS transport ready");
	return 0;
}
