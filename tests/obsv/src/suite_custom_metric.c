/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file suite_custom_metric.c
 * @brief Tests for application-defined metrics (not the stock probs / TM descriptors).
 */

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/ztest.h>

#include "common.h"

/** Outside the built-in metric ID range. */
#define CUSTOM_METRIC_ID  100U
#define CUSTOM_METRIC_VER 1U

static nrf_edgeai_obsv_core_t ctx;

static const nrf_edgeai_obsv_model_info_t test_model = {
	.model_id = TEST_MODEL_ID,
	.num_classes = TEST_NUM_CLASSES,
	.version = TEST_MODEL_VERSION,
};

static uint32_t custom_counter;
static uint32_t custom_shadow[1];

static void custom_init(const void *cfg, void *priv)
{
	ARG_UNUSED(cfg);
	ARG_UNUSED(priv);
	custom_counter = 0U;
}

static void custom_clear(void *priv)
{
	ARG_UNUSED(priv);
	custom_counter = 0U;
}

static void custom_update(const float *p_probs, uint16_t n, void *priv)
{
	ARG_UNUSED(p_probs);
	ARG_UNUSED(n);
	ARG_UNUSED(priv);
	custom_counter++;
}

static void custom_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	ARG_UNUSED(priv);
	custom_shadow[0] = custom_counter;

	out->metric_id = CUSTOM_METRIC_ID;
	out->version = CUSTOM_METRIC_VER;
	out->num_rows = 1U;
	out->num_cols = 1U;
	out->counts = custom_shadow;
}

static nrf_edgeai_obsv_metric_t custom_metric = {
	.init = custom_init,
	.update = custom_update,
	.clear = custom_clear,
	.finalize = NULL,
	.snapshot = custom_snapshot,
	.priv = NULL,
	.p_next = NULL,
};

/* Built-in metric instance for this test suite. */
static uint32_t custom_suite_pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(TEST_NUM_CLASSES) /
				    sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t custom_suite_pd;

static void custom_before(void *fixture)
{
	ARG_UNUSED(fixture);
	memset(&ctx, 0, sizeof(ctx));
	nrf_edgeai_obsv_core_init(&ctx, &test_model);
	nrf_edgeai_obsv_metric_pd_create(&custom_suite_pd, custom_suite_pd_buf, TEST_NUM_CLASSES);
}

ZTEST_SUITE(obsv_custom_metric, NULL, NULL, custom_before, NULL, NULL);

struct verify_custom_user {
	bool ok;
};

static bool verify_custom_snap_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct verify_custom_user *v = user;

	zassert_equal(snap->metric_id, CUSTOM_METRIC_ID);
	zassert_equal(snap->version, CUSTOM_METRIC_VER);
	zassert_equal(snap->num_rows, 1U);
	zassert_equal(snap->num_cols, 1U);
	zassert_equal(snap->counts[0], 2U);
	v->ok = true;
	return true;
}

ZTEST(obsv_custom_metric, test_register_update_snapshot)
{
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &custom_metric, NULL), 0);
	zassert_equal(ctx.num_metrics, 1U);

	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	nrf_edgeai_obsv_core_update(&ctx, probs);
	nrf_edgeai_obsv_core_update(&ctx, probs);

	zassert_equal(ctx.num_inferences, 2U);

	struct verify_custom_user vu = {.ok = false};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, verify_custom_snap_cb, &vu), 0);
	zassert_true(vu.ok, "custom metric snapshot not observed");
}

struct dual_verify {
	bool saw_probs;
	bool saw_custom;
};

static bool dual_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct dual_verify *v = user;

	if (snap->metric_id == NRF_EDGEAI_OBSV_METRIC_ID_PROBS_DISTRIBUTION) {
		v->saw_probs = true;
	} else if (snap->metric_id == CUSTOM_METRIC_ID) {
		zassert_equal(snap->counts[0], 1U);
		v->saw_custom = true;
	}
	return true;
}

ZTEST(obsv_custom_metric, test_custom_alongside_builtin)
{
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &custom_suite_pd, NULL), 0);
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &custom_metric, NULL), 0);
	zassert_equal(ctx.num_metrics, 2U);

	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	nrf_edgeai_obsv_core_update(&ctx, probs);

	zassert_equal(ctx.num_inferences, 1U);

	struct dual_verify dv = {
		.saw_probs = false,
		.saw_custom = false,
	};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, dual_capture_cb, &dv), 0);
	zassert_true(dv.saw_probs, "built-in probs metric missing from traversal");
	zassert_true(dv.saw_custom, "custom metric missing from traversal");
}

static bool read_custom_snap_after_reset_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap,
					    void *user)
{
	bool *zero_snap = user;

	if (snap->metric_id == CUSTOM_METRIC_ID) {
		*zero_snap = (snap->counts[0] == 0U);
	}
	return true;
}

ZTEST(obsv_custom_metric, test_reset_reinitializes_custom)
{
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &custom_metric, NULL), 0);

	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	nrf_edgeai_obsv_core_update(&ctx, probs);
	zassert_equal(custom_counter, 1U);

	zassert_equal(nrf_edgeai_obsv_core_reset(&ctx), 0);

	bool zero_snap = false;
	int fe_ret = nrf_edgeai_obsv_core_for_each_metric(&ctx, read_custom_snap_after_reset_cb,
							  &zero_snap);

	zassert_equal(fe_ret, 0);
	zassert_true(zero_snap, "reset did not re-init custom metric counters");
}
