/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_hid.h"

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>
#include <zephyr/settings/settings.h>

#include <bluetooth/services/hids.h>
#include "ux_state_manager.h"

LOG_MODULE_REGISTER(ble_hid);

/* HID keyboard usage codes */
#define KEY_ARROW_LEFT	0x50
#define KEY_ARROW_RIGHT 0x4F
#define KEY_F5		0x3E
#define KEY_ESC		0x29

/* Consumer control bit positions (1-byte bitmap) */
#define KEY_MEDIA_VOLUME_UP   BIT(0)
#define KEY_MEDIA_VOLUME_DOWN BIT(1)
#define KEY_MEDIA_MUTE	      BIT(2)
#define KEY_MEDIA_PLAY_PAUSE  BIT(3)
#define KEY_MEDIA_PREV_TRACK  BIT(4)
#define KEY_MEDIA_NEXT_TRACK  BIT(5)

/* Input report indices within the report group */
#define INPUT_REP_KEYBOARD_IDX 0
#define INPUT_REP_CONSUMER_IDX 1

/* Report IDs matching the report map */
#define INPUT_REP_KEYBOARD_ID 0x01
#define INPUT_REP_CONSUMER_ID 0x02

/* Report sizes in bytes */
#define INPUT_REP_KEYBOARD_LEN 8
#define INPUT_REP_CONSUMER_LEN 1

/* Offset of the 'key' byte within the HID report array */
#define INPUT_REP_KEYBOARD_KEY_OFFSET 2
#define INPUT_REP_CONSUMER_KEY_OFFSET 0

/* Advertising retry configuration */
#define ADV_RETRY_MAX     10
#define ADV_RETRY_PERIOD  K_MSEC(500)

#define BASE_USB_HID_SPEC_VERSION 0x0101

BT_HIDS_DEF(hids_obj, INPUT_REP_KEYBOARD_LEN, INPUT_REP_CONSUMER_LEN);

struct pairing_data_mitm {
	struct bt_conn *conn;
	unsigned int passkey;
};

/* Advertising related symbols */
static void adv_work_handler(struct k_work *work);
static K_WORK_DELAYABLE_DEFINE(adv_work, adv_work_handler);
static atomic_t adv_retry_count;

K_MSGQ_DEFINE(mitm_queue, sizeof(struct pairing_data_mitm), 1, 4);
static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID16_ALL, BT_UUID_16_ENCODE(BT_UUID_HIDS_VAL),
		      BT_UUID_16_ENCODE(BT_UUID_BAS_VAL)),
	BT_DATA_BYTES(BT_DATA_GAP_APPEARANCE,
		      BT_BYTES_LIST_LE16(BT_APPEARANCE_HID_PRESENTATION_REMOTE)),
};

static const struct bt_data sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, CONFIG_BT_DEVICE_NAME, sizeof(CONFIG_BT_DEVICE_NAME) - 1),
};

static const uint8_t report_map[] = {
	0x05, 0x01, /* Usage Page (Generic Desktop) */
	0x09, 0x06, /* Usage (Keyboard) */
	0xA1, 0x01, /* Collection (Application) */

	0x05, 0x07, /*   Usage Page (Keyboard/Keypad) */
	0x85, 0x01, /*   Report ID (1) */
	0x19, 0xE0, /*   Usage Minimum (Keyboard Left Control) */
	0x29, 0xE7, /*   Usage Maximum (Keyboard Right GUI) */
	0x15, 0x00, /*   Logical Minimum (0) */
	0x25, 0x01, /*   Logical Maximum (1) */
	0x75, 0x01, /*   Report Size (1) */
	0x95, 0x08, /*   Report Count (8) */
	0x81, 0x02, /*   Input (Data, Variable, Absolute) ; Modifier keys */

	0x95, 0x01, /*   Report Count (1) */
	0x75, 0x08, /*   Report Size (8) */
	0x81, 0x01, /*   Input (Constant) ; Reserved byte */

	0x95, 0x05, /*   Report Count (5) */
	0x75, 0x01, /*   Report Size (1) */
	0x05, 0x08, /*   Usage Page (LEDs) */
	0x85, 0x01, /*   Report ID (1) */
	0x19, 0x01, /*   Usage Minimum (Num Lock) */
	0x29, 0x05, /*   Usage Maximum (Kana) */
	0x91, 0x02, /*   Output (Data, Variable, Absolute) ; LED report */

	0x95, 0x01, /*   Report Count (1) */
	0x75, 0x03, /*   Report Size (3) */
	0x91, 0x03, /*   Output (Constant) ; LED report padding */

	0x95, 0x06, /*   Report Count (6) */
	0x75, 0x08, /*   Report Size (8) */
	0x15, 0x00, /*   Logical Minimum (0) */
	0x25, 0x65, /*   Logical Maximum (101) */
	0x05, 0x07, /*   Usage Page (Keyboard/Keypad) */
	0x19, 0x00, /*   Usage Minimum (Reserved (no event indicated)) */
	0x29, 0x65, /*   Usage Maximum (Keyboard Application) */
	0x81, 0x00, /*   Input (Data, Array) ; Key arrays (6 bytes) */

	0xC0, /* End Collection */

	/* Consumer Control Report */
	0x05, 0x0C, /* Usage Page (Consumer Devices) */
	0x09, 0x01, /* Usage (Consumer Control) */
	0xA1, 0x01, /* Collection (Application) */

	0x85, 0x02, /*   Report ID (2) */
	0x05, 0x0C, /*   Usage Page (Consumer Devices) */
	0x15, 0x00, /*   Logical Minimum (0) */
	0x25, 0x01, /*   Logical Maximum (1) */

	0x09, 0xE9, /*   Usage (Volume Up) */
	0x09, 0xEA, /*   Usage (Volume Down) */
	0x09, 0xE2, /*   Usage (Mute) */
	0x09, 0xCD, /*   Usage (Play/Pause) */
	0x19, 0xB5, /*   Usage Minimum (Scan Next Track) */
	0x29, 0xB8, /*   Usage Maximum (Scan Previous Track) */

	0x75, 0x01, /*   Report Size (1) */
	0x95, 0x08, /*   Report Count (8) */
	0x81, 0x02, /*   Input (Data, Variable, Absolute) ; Media keys */

	0xC0 /* End Collection */
};

/* HID notifications are only usable once the peer has subscribed by writing
 * the CCCD of the corresponding input report. Until then bt_hids_inp_rep_send()
 * fails (e.g. -ENOTSUP/-ENODATA) even though we are connected and encrypted.
 * Track per-report subscription state so we can silently skip sending until
 * the BLE stack is fully ready to deliver notifications.
 */
static bool kb_notif_enabled;
static bool consumer_notif_enabled;

static void kb_notif_handler(enum bt_hids_notify_evt evt)
{
	kb_notif_enabled = (evt == BT_HIDS_CCCD_EVT_NOTIFY_ENABLED);
	LOG_DBG("Keyboard notifications %s", kb_notif_enabled ? "enabled" : "disabled");
}

static void consumer_notif_handler(enum bt_hids_notify_evt evt)
{
	consumer_notif_enabled = (evt == BT_HIDS_CCCD_EVT_NOTIFY_ENABLED);
	LOG_DBG("Consumer notifications %s", consumer_notif_enabled ? "enabled" : "disabled");
}

static void adv_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);

	int err = bt_le_adv_start(BT_LE_ADV_CONN_FAST_1, ad, ARRAY_SIZE(ad),
				  sd, ARRAY_SIZE(sd));

	if (err == 0) {
		int retries = atomic_get(&adv_retry_count);

		atomic_set(&adv_retry_count, 0);

		LOG_INF("Advertising successfully started%s",
			retries ? " (after retry)" : "");

		return;
	}

	if (err == -EALREADY) {
		LOG_DBG("Advertising already running");
		atomic_set(&adv_retry_count, 0);
		return;
	}

	atomic_inc(&adv_retry_count);

	int adv_retry_count_val = atomic_get(&adv_retry_count);

	if (adv_retry_count_val < ADV_RETRY_MAX) {
		LOG_WRN("Advertising start failed (err %d), retry %d/%d in 500 ms",
			err, adv_retry_count_val, ADV_RETRY_MAX);
		k_work_reschedule(&adv_work, ADV_RETRY_PERIOD);
	} else {
		LOG_ERR("Advertising failed to start after %d attempts (err %d)",
			ADV_RETRY_MAX, err);
		atomic_set(&adv_retry_count, 0);
	}
}

static int start_advertising(void)
{
	/* Kick off (or reset) the asynchronous advertising-start sequence.
	 * Actual bt_le_adv_start() is invoked from the system workqueue and
	 * will retry up to ADV_RETRY_MAX times with ADV_RETRY_PERIOD spacing.
	 */
	atomic_set(&adv_retry_count, 0);
	(void)k_work_reschedule(&adv_work, K_NO_WAIT);
	return 0;
}

static void connected(struct bt_conn *conn, uint8_t conn_err)
{
	char addr[BT_ADDR_LE_STR_LEN];
	bt_security_t required_level;
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (conn_err) {
		LOG_ERR("Failed to connect to %s (%u)", addr, conn_err);
		return;
	}

	LOG_INF("Connected %s", addr);

	err = bt_hids_connected(&hids_obj, conn);
	if (err) {
		LOG_ERR("Failed to notify HIDS about connection (err %d)", err);
	}

	/* Application security policy:
	 * - MITM enabled  -> authenticated LE Secure Connections (L4)
	 * - MITM disabled -> encrypted link without authentication (L2)
	 */
	required_level = IS_ENABLED(CONFIG_BLE_MITM_AUTH) ? BT_SECURITY_L4 : BT_SECURITY_L2;

	err = bt_conn_set_security(conn, required_level);
	if (err) {
		LOG_ERR("Failed to request security level %d (err %d), disconnecting %s",
			required_level, err, addr);

		err = bt_conn_disconnect(conn, BT_HCI_ERR_AUTH_FAIL);
		if (err && err != -ENOTCONN) {
			LOG_ERR("Failed to disconnect %s (err %d)", addr, err);
		}
		/* Do not mark as connected or notify the app; wait for the
		 * disconnected() callback which will restart advertising.
		 */
		return;
	}

	ble_common_set_connected(true);
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_INF("Disconnected from %s (reason 0x%02x)", addr, reason);

	kb_notif_enabled = false;
	consumer_notif_enabled = false;

	err = bt_hids_disconnected(&hids_obj, conn);
	if (err) {
		LOG_ERR("Failed to notify HIDS about disconnection (err %d)", err);
	}

	ble_common_set_connected(false);

	/* If disconnected due to authentication failure, clear connection pairing info */
	if (reason == BT_HCI_ERR_AUTH_FAIL || reason == BT_HCI_ERR_PIN_OR_KEY_MISSING) {
		LOG_INF("Authentication related disconnect, clearing pairing info");
		bt_unpair(BT_ID_DEFAULT, bt_conn_get_dst(conn));
	}

	/* Defensively get back to normal button function as we are not in pairing mode anymore */
	uxsm_set_btn_func(UX_BTN_FUNC_NORMAL);

	/* Restart advertising */
	start_advertising();
}

static void security_changed(struct bt_conn *conn, bt_security_t level, enum bt_security_err err)
{
	char addr[BT_ADDR_LE_STR_LEN];
	int disc_err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	if (!err) {
		LOG_INF("Security changed: %s level %u", addr, level);
	} else {
		LOG_ERR("Security failed: %s level %u err %d", addr, level, err);
		bt_unpair(BT_ID_DEFAULT, bt_conn_get_dst(conn));

		/* The link is up but does not meet the required security level,
		 * so force-disconnect. The disconnected() callback will restart
		 * advertising.
		 */
		disc_err = bt_conn_disconnect(conn, BT_HCI_ERR_AUTH_FAIL);
		if (disc_err && disc_err != -ENOTCONN) {
			LOG_ERR("Failed to disconnect after security failure (err %d)",
				disc_err);
		}
	}
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
	.security_changed = security_changed,
};

static int hids_service_init(void)
{
	struct bt_hids_init_param hids_init_param = {0};
	struct bt_hids_inp_rep *keyboard_rep;
	struct bt_hids_inp_rep *consumer_rep;

	hids_init_param.rep_map.data = report_map;
	hids_init_param.rep_map.size = sizeof(report_map);

	hids_init_param.info.bcd_hid = BASE_USB_HID_SPEC_VERSION;
	hids_init_param.info.b_country_code = 0x00;
	hids_init_param.info.flags = BT_HIDS_NORMALLY_CONNECTABLE;

	keyboard_rep = &hids_init_param.inp_rep_group_init.reports[INPUT_REP_KEYBOARD_IDX];
	keyboard_rep->size = INPUT_REP_KEYBOARD_LEN;
	keyboard_rep->id = INPUT_REP_KEYBOARD_ID;
	keyboard_rep->handler = kb_notif_handler;
	hids_init_param.inp_rep_group_init.cnt++;

	consumer_rep = &hids_init_param.inp_rep_group_init.reports[INPUT_REP_CONSUMER_IDX];
	consumer_rep->size = INPUT_REP_CONSUMER_LEN;
	consumer_rep->id = INPUT_REP_CONSUMER_ID;
	consumer_rep->handler = consumer_notif_handler;
	hids_init_param.inp_rep_group_init.cnt++;

	hids_init_param.is_kb = true;

	return bt_hids_init(&hids_obj, &hids_init_param);
}

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

	start_advertising();
}

#if IS_ENABLED(CONFIG_BLE_MITM_AUTH)
static void auth_passkey_display(struct bt_conn *conn, unsigned int passkey)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_INF("Passkey for %s: %06u", addr, passkey);
}

static void auth_passkey_confirm(struct bt_conn *conn, unsigned int passkey)
{
	int err;
	char addr[BT_ADDR_LE_STR_LEN];
	struct pairing_data_mitm pairing_data;

	pairing_data.conn = bt_conn_ref(conn);
	pairing_data.passkey = passkey;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	err = k_msgq_put(&mitm_queue, &pairing_data, K_NO_WAIT);

	if (err) {
		LOG_ERR("Failed to queue pairing confirmation for %s (err %d)", addr, err);
		bt_conn_unref(pairing_data.conn);
		bt_conn_auth_cancel(conn);
		return;
	}

	LOG_INF("Passkey for %s: %06u", addr, pairing_data.passkey);

	/* Switch the application to pairing state. The state manager will print
	 * user-facing instructions about which press confirms / rejects pairing.
	 */
	uxsm_set_btn_func(UX_BTN_FUNC_PAIRING);
}
#endif /* CONFIG_BLE_MITM_AUTH */

static void auth_cancel(struct bt_conn *conn)
{
	char addr[BT_ADDR_LE_STR_LEN];
	struct pairing_data_mitm pairing_data;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Pairing cancelled: %s", addr);

	/* If we had queued a numeric-comparison request for this conn,
	 * drop it and release the ref we took in auth_passkey_confirm().
	 */
	if (k_msgq_peek(&mitm_queue, &pairing_data) == 0 &&
	    pairing_data.conn == conn) {
		(void)k_msgq_get(&mitm_queue, &pairing_data, K_NO_WAIT);
		bt_conn_unref(pairing_data.conn);
	}

	uxsm_set_btn_func(UX_BTN_FUNC_NORMAL);
}

static void pairing_complete(struct bt_conn *conn, bool bonded)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_INF("Pairing completed: %s, bonded: %d", addr, bonded);
	uxsm_set_btn_func(UX_BTN_FUNC_NORMAL);
}

static void pairing_failed(struct bt_conn *conn, enum bt_security_err reason)
{
	char addr[BT_ADDR_LE_STR_LEN];
	struct pairing_data_mitm pairing_data;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	LOG_ERR("Pairing failed conn: %s, reason %d %s", addr, reason,
		bt_security_err_to_str(reason));

	if (k_msgq_peek(&mitm_queue, &pairing_data) == 0 && pairing_data.conn == conn) {
		(void)k_msgq_get(&mitm_queue, &pairing_data, K_NO_WAIT);
		bt_conn_unref(pairing_data.conn);
	}

	uxsm_set_btn_func(UX_BTN_FUNC_NORMAL);
}

static struct bt_conn_auth_cb conn_auth_callbacks = {
#if IS_ENABLED(CONFIG_BLE_MITM_AUTH)
	/* With MITM protection enabled, advertise DisplayYesNo I/O capabilities
	 * so the stack negotiates Numeric Comparison (authenticated pairing).
	 */
	.passkey_display = auth_passkey_display,
	.passkey_confirm = auth_passkey_confirm,
#endif
	.cancel = auth_cancel,
};

static struct bt_conn_auth_info_cb conn_auth_info_callbacks = {.pairing_complete = pairing_complete,
							       .pairing_failed = pairing_failed};

static void disconnect_cb(struct bt_conn *conn, void *data)
{
	int *err_out = data;
	char addr[BT_ADDR_LE_STR_LEN];
	int err;

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));
	LOG_INF("Disconnecting %s", addr);

	err = bt_conn_disconnect(conn, BT_HCI_ERR_REMOTE_USER_TERM_CONN);
	if (err && err != -ENOTCONN) {
		LOG_ERR("Failed to disconnect %s (err %d)", addr, err);
		*err_out = err;
	}
}

int ble_hid_confirm_pairing(bool accept)
{
	struct pairing_data_mitm pairing_data;

	int err = k_msgq_get(&mitm_queue, &pairing_data, K_NO_WAIT);

	if (err) {
		return err;
	}

	if (accept) {
		err = bt_conn_auth_passkey_confirm(pairing_data.conn);
		if (err) {
			LOG_ERR("Failed to confirm pairing (err %d)", err);
		} else {
			LOG_INF("Numeric Match, conn %p", pairing_data.conn);
		}
	} else {
		err = bt_conn_auth_cancel(pairing_data.conn);

		if (err) {
			LOG_ERR("Failed to reject pairing (err %d)", err);
		} else {
			LOG_INF("Numeric Reject, conn %p", pairing_data.conn);
		}
	}
	bt_conn_unref(pairing_data.conn);

	return err;
}

int ble_hid_forget_bonds(void)
{
	int disc_err = 0;
	int err;

	/* Disconnect all active connections.
	 * The disconnected() callback will call start_advertising() once each
	 * connection tears down.
	 */
	bt_conn_foreach(BT_CONN_TYPE_LE, disconnect_cb, &disc_err);

	/* Remove all local bond records */
	LOG_INF("Clearing all bonding information");
	err = bt_unpair(BT_ID_DEFAULT, BT_ADDR_LE_ANY);
	if (err) {
		LOG_ERR("Failed to unpair (err %d)", err);
		return err;
	}

	return disc_err;
}

int ble_hid_init(void)
{
	int err;

	err = bt_conn_auth_cb_register(&conn_auth_callbacks);
	if (err) {
		LOG_ERR("Failed to register authorization callbacks.");
		return err;
	}

	err = bt_conn_auth_info_cb_register(&conn_auth_info_callbacks);
	if (err) {
		LOG_ERR("Failed to register authorization info callbacks.");
		return err;
	}

	err = hids_service_init();

	if (err) {
		LOG_ERR("HID Service init failed (err %d)", err);
		return err;
	}

	err = bt_enable(bt_ready);

	if (err) {
		LOG_ERR("Bluetooth init failed (err %d)", err);
		return err;
	}

	return err;
}

int ble_hid_send_key(ble_hid_key_t key)
{
	if (uxsm_get_btn_func() == UX_BTN_FUNC_PAIRING) {
		LOG_WRN("Currently in pairing mode, not sending key");
		return -EBUSY;
	}
	int err;
	uint8_t keyboard_report[INPUT_REP_KEYBOARD_LEN] = {0};
	uint8_t consumer_report[INPUT_REP_CONSUMER_LEN] = {0};
	bool is_consumer = false;

	if (!ble_common_is_connected()) {
		return -ENOTCONN;
	}

	/* The keyboard report is sent as a single key pressed with modifier key bitmap cleared */
	switch (key) {
	case BLE_HID_KEY_ARROW_LEFT:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_ARROW_LEFT;
		break;
	case BLE_HID_KEY_ARROW_RIGHT:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_ARROW_RIGHT;
		break;
	case BLE_HID_KEY_F5:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_F5;
		break;
	case BLE_HID_KEY_ESC:
		keyboard_report[INPUT_REP_KEYBOARD_KEY_OFFSET] = KEY_ESC;
		break;
	case BLE_HID_KEY_MEDIA_PREV_TRACK:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_PREV_TRACK;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_NEXT_TRACK:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_NEXT_TRACK;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_MUTE:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_MUTE;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_PLAY_PAUSE:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_PLAY_PAUSE;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_VOLUME_UP:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_VOLUME_UP;
		is_consumer = true;
		break;
	case BLE_HID_KEY_MEDIA_VOLUME_DOWN:
		consumer_report[INPUT_REP_CONSUMER_KEY_OFFSET] = KEY_MEDIA_VOLUME_DOWN;
		is_consumer = true;
		break;
	default:
		return -EINVAL;
	}

	/* The peer subscribes to HID input notifications by writing the
	 * corresponding CCCD some time after the link is encrypted. Calling
	 * bt_hids_inp_rep_send() before that point fails with errors such as
	 * -ENOTSUP / -ENODATA. Skip silently until the BLE stack is ready.
	 */
	if (is_consumer ? !consumer_notif_enabled : !kb_notif_enabled) {
		LOG_WRN("Peer has not subscribed to HID input notifications yet "
			"(CCCD not written); dropping key");
		return -EAGAIN;
	}

	if (is_consumer) {
		err = bt_hids_inp_rep_send(&hids_obj, NULL, INPUT_REP_CONSUMER_IDX, consumer_report,
					   sizeof(consumer_report), NULL);
		if (err) {
			LOG_ERR("Failed to send consumer key (err %d)", err);
			return err;
		}

		memset(consumer_report, 0, sizeof(consumer_report));
		err = bt_hids_inp_rep_send(&hids_obj, NULL, INPUT_REP_CONSUMER_IDX, consumer_report,
					   sizeof(consumer_report), NULL);
	} else {
		err = bt_hids_inp_rep_send(&hids_obj, NULL, INPUT_REP_KEYBOARD_IDX, keyboard_report,
					   sizeof(keyboard_report), NULL);
		if (err) {
			LOG_ERR("Failed to send keyboard key (err %d)", err);
			return err;
		}

		memset(keyboard_report, 0, sizeof(keyboard_report));
		err = bt_hids_inp_rep_send(&hids_obj, NULL, INPUT_REP_KEYBOARD_IDX, keyboard_report,
					   sizeof(keyboard_report), NULL);
	}

	if (err) {
		LOG_ERR("Failed to send key release (err %d)", err);
		return err;
	}

	LOG_INF("BLE HID key sent successfully");

	return 0;
}
