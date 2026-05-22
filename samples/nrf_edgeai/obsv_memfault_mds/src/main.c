/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file main.c
 * @brief nRF Edge AI Observability over Memfault MDS sample.
 *
 * Feeds two synthetic probability streams with different class counts into the
 * Edge AI observability library under two separate model IDs, lets the Memfault
 * CDR glue snapshot both every 60 s as a single CDR (CBOR array with two
 * entries), and exposes the Memfault Diagnostic Service (MDS) over BLE so a
 * phone / gateway can drain the payload into a Memfault project.
 *
 * Model A uses NUM_CLASSES_A output classes; model B uses NUM_CLASSES_B.
 * Having different class counts exercises the library's per-context sizing.
 *
 * No real model runs here - the focus is on the multi-model observability
 * pipeline and the on-device Memfault plumbing.
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/settings/settings.h>
#include <zephyr/sys/util.h>

#include <bluetooth/services/mds.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

LOG_MODULE_REGISTER(stats_mds_sample, LOG_LEVEL_DBG);

#define DEVICE_NAME	CONFIG_BT_DEVICE_NAME
#define DEVICE_NAME_LEN (sizeof(DEVICE_NAME) - 1)

#define NUM_CLASSES_A	      4 /* gesture-like: 4 output classes */
#define NUM_CLASSES_B	      3 /* anomaly-like: 3 output classes */
#define SAMPLE_MODEL_ID_A     0x5354 /* "ST" for SynTh model A. */
#define SAMPLE_MODEL_ID_B     0x414E /* "AN" for ANomaly-like model B. */
#define SAMPLE_MODEL_VERSION  1

BUILD_ASSERT(NUM_CLASSES_A <= CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES,
	     "Model A class count exceeds observability library limit");
BUILD_ASSERT(NUM_CLASSES_B <= CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES,
	     "Model B class count exceeds observability library limit");

/*
 * Two independent observability contexts, one per synthetic model.
 * Both are registered with the Memfault transport; the auto-collect worker
 * encodes them into a single CDR as a two-element CBOR array.
 *
 * Model A rotates its dominant class at the Kconfig-defined period.
 * Model B rotates at half that period so the two models produce visually
 * distinct transition matrices in the decoded CDR.
 *
 * Each metric uses caller-provided uint32_t-aligned storage so model A and B
 * keep fully independent counters.
 */
static nrf_edgeai_obsv_ctx_t obsv_ctx_a;
static nrf_edgeai_obsv_ctx_t obsv_ctx_b;

static uint32_t pd_a_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(NUM_CLASSES_A) / sizeof(uint32_t)];
static uint32_t tm_a_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(NUM_CLASSES_A) / sizeof(uint32_t)];
static uint32_t pd_b_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(NUM_CLASSES_B) / sizeof(uint32_t)];
static uint32_t tm_b_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(NUM_CLASSES_B) / sizeof(uint32_t)];

static nrf_edgeai_obsv_metric_t pd_a;
static nrf_edgeai_obsv_metric_t tm_a;
static nrf_edgeai_obsv_metric_t pd_b;
static nrf_edgeai_obsv_metric_t tm_b;

/* MDS advertising. UUID128_ALL advertising data lets the Memfault phone app
 * and gateway discover the device without a scan filter on the name.
 */
static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID128_ALL, BT_UUID_MDS_VAL),
};

static const struct bt_data sd[] = {
	BT_DATA(BT_DATA_NAME_COMPLETE, DEVICE_NAME, DEVICE_NAME_LEN),
};

static struct bt_conn *mds_conn;
static struct k_work adv_work;

static void inference_work_handler(struct k_work *work);
static K_WORK_DELAYABLE_DEFINE(inference_work, inference_work_handler);

/* -------- Synthetic 4-class data generator --------
 *
 * Produces a probability vector where one "dominant" class carries a high
 * mass (~0.70) and the remaining mass is split across the other classes
 * with small random perturbations. The dominant class rotates through
 * 0 -> 1 -> 2 -> 3 every @p rot_period_ms milliseconds so downstream sees
 * a non-trivial transition matrix and per-class probability distributions.
 */
static void synth_generate(float *probs, uint8_t n_classes, uint32_t rot_period_ms)
{
	const uint32_t now_ms = k_uptime_get_32();
	const uint8_t dominant = (now_ms / rot_period_ms) % n_classes;

	const float dominant_prob = 0.70f;
	const float other_prob = (1.0f - dominant_prob) / (n_classes - 1);

	/* Use a tiny deterministic LCG for jitter. We don't need cryptographic
	 * quality here and avoiding the Zephyr RNG subsystem keeps the sample
	 * self-contained.
	 */
	static uint32_t lcg_state = 0xA5A5A5A5u;
	const float jitter_span = 0.10f;

	float sum = 0.0f;

	for (uint8_t i = 0; i < n_classes; i++) {
		lcg_state = lcg_state * 1664525u + 1013904223u;
		uint32_t r = lcg_state >> 8;
		float jitter = ((float)(r % 1001) / 1000.0f - 0.5f) * jitter_span;

		probs[i] = (i == dominant ? dominant_prob : other_prob) + jitter;
		if (probs[i] < 0.0f) {
			probs[i] = 0.0f;
		}
		sum += probs[i];
	}

	/* Renormalize to keep the observability metrics semantically meaningful
	 * (sum(probs) == 1). A zero-sum is impossible here because the
	 * dominant class baseline is well above the jitter span.
	 */
	for (uint8_t i = 0; i < n_classes; i++) {
		probs[i] /= sum;
	}
}

static void inference_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);

	static uint32_t tick;

	const uint32_t period_a_ms = CONFIG_NRF_EDGEAI_OBSV_SAMPLE_ROTATION_PERIOD_SEC * 1000U;
	const uint32_t period_b_ms = period_a_ms / 2U; /* model B rotates 2x faster */

	float probs_a[NUM_CLASSES_A];
	float probs_b[NUM_CLASSES_B];

	synth_generate(probs_a, NUM_CLASSES_A, period_a_ms);
	if (nrf_edgeai_obsv_update(&obsv_ctx_a, probs_a)) {
		LOG_WRN("obsv_a update failed");
	}

	synth_generate(probs_b, NUM_CLASSES_B, period_b_ms);
	if (nrf_edgeai_obsv_update(&obsv_ctx_b, probs_b)) {
		LOG_WRN("obsv_b update failed");
	}

	/* Log every 5th tick so UART is not spammed. */
	if ((tick++ % 5) == 0) {
		const uint8_t dom_a = (k_uptime_get_32() / period_a_ms) % NUM_CLASSES_A;
		const uint8_t dom_b = (k_uptime_get_32() / period_b_ms) % NUM_CLASSES_B;

		LOG_DBG("tick %u  A dom=%u  B dom=%u", tick, dom_a, dom_b);
	}

	(void)k_work_reschedule(&inference_work,
				K_MSEC(CONFIG_NRF_EDGEAI_OBSV_SAMPLE_INFERENCE_PERIOD_MS));
}

/* -------- Bluetooth + MDS plumbing --------
 *
 * Cribbed from samples/bluetooth/peripheral_mds. MDS requires an encrypted
 * link before the phone is granted access to diagnostic data.
 */
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
	if (conn_err) {
		LOG_ERR("Connection failed, err 0x%02x", conn_err);
		return;
	}

	LOG_INF("Connected");
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	LOG_INF("Disconnected, reason 0x%02x", reason);

	if (conn == mds_conn) {
		mds_conn = NULL;
	}
}

static void recycled_cb(void)
{
	advertising_start();
}

static void security_changed(struct bt_conn *conn, bt_security_t level, enum bt_security_err err)
{
	if (err) {
		LOG_WRN("Security failed: level %u err %d", level, err);
		return;
	}

	LOG_INF("Security changed: level %u", level);

	if (level >= BT_SECURITY_L2 && !mds_conn) {
		mds_conn = conn;
	}
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

static int bt_setup(void)
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

int main(void)
{
	const nrf_edgeai_obsv_model_info_t model_a = {
		.model_id = SAMPLE_MODEL_ID_A,
		.num_classes = NUM_CLASSES_A,
		.version = SAMPLE_MODEL_VERSION,
	};
	const nrf_edgeai_obsv_model_info_t model_b = {
		.model_id = SAMPLE_MODEL_ID_B,
		.num_classes = NUM_CLASSES_B,
		.version = SAMPLE_MODEL_VERSION,
	};
	int err;

	LOG_INF("Starting nRF Edge AI Observability over Memfault MDS sample (2 models)");

	err = nrf_edgeai_obsv_init(&obsv_ctx_a, &model_a);
	if (err) {
		LOG_ERR("obsv_a init failed: %d", err);
		return 0;
	}

	nrf_edgeai_obsv_metric_pd_create(&pd_a, pd_a_buf, NUM_CLASSES_A);
	nrf_edgeai_obsv_metric_tm_create(&tm_a, tm_a_buf, NUM_CLASSES_A);

	err = nrf_edgeai_obsv_register(&obsv_ctx_a, &pd_a, NULL);
	if (err) {
		LOG_ERR("obsv_a: probs distribution register failed: %d", err);
		return 0;
	}

	err = nrf_edgeai_obsv_register(&obsv_ctx_a, &tm_a, NULL);
	if (err) {
		LOG_ERR("obsv_a: transition matrix register failed: %d", err);
		return 0;
	}

	err = nrf_edgeai_obsv_memfault_init(&obsv_ctx_a);
	if (err) {
		LOG_ERR("obsv_a: memfault init failed: %d", err);
		return 0;
	}

	err = nrf_edgeai_obsv_init(&obsv_ctx_b, &model_b);
	if (err) {
		LOG_ERR("obsv_b init failed: %d", err);
		return 0;
	}

	nrf_edgeai_obsv_metric_pd_create(&pd_b, pd_b_buf, NUM_CLASSES_B);
	nrf_edgeai_obsv_metric_tm_create(&tm_b, tm_b_buf, NUM_CLASSES_B);

	err = nrf_edgeai_obsv_register(&obsv_ctx_b, &pd_b, NULL);
	if (err) {
		LOG_ERR("obsv_b: probs distribution register failed: %d", err);
		return 0;
	}

	err = nrf_edgeai_obsv_register(&obsv_ctx_b, &tm_b, NULL);
	if (err) {
		LOG_ERR("obsv_b: transition matrix register failed: %d", err);
		return 0;
	}

	err = nrf_edgeai_obsv_memfault_init(&obsv_ctx_b);
	if (err) {
		LOG_ERR("obsv_b: memfault init failed: %d", err);
		return 0;
	}

	err = bt_setup();
	if (err) {
		return 0;
	}

	LOG_INF("Bluetooth initialized");

	/* Kick the synthetic inference loop. The Memfault CDR glue refreshes
	 * its snapshot independently every
	 * CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC seconds.
	 */
	(void)k_work_reschedule(&inference_work,
				K_MSEC(CONFIG_NRF_EDGEAI_OBSV_SAMPLE_INFERENCE_PERIOD_MS));

	return 0;
}
