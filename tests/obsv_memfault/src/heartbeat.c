/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <string.h>

#include <zephyr/ztest.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>

#define TEST_ROWS 3U
#define TEST_COLS 4U

void memfault_mock_reset(void);
void memfault_mock_set_storage_full(bool full);
uint32_t memfault_mock_value(int key);
uint32_t memfault_mock_write_count(void);

static uint32_t counts[TEST_ROWS * TEST_COLS];

static void metric_init(const void *cfg, void *priv)
{
	ARG_UNUSED(cfg);
	memset(priv, 0, sizeof(counts));
}

static void metric_update(const float *data, uint16_t n, void *priv)
{
	ARG_UNUSED(data);
	ARG_UNUSED(n);

	uint32_t *values = priv;

	values[0]++;
	values[TEST_COLS + 2U] += 7U;
}

static void metric_clear(void *priv)
{
	memset(priv, 0, sizeof(counts));
}

static void metric_snapshot(nrf_edgeai_obsv_metric_snapshot_t *snapshot, void *priv)
{
	*snapshot = (nrf_edgeai_obsv_metric_snapshot_t){
		.metric_id = NRF_EDGEAI_OBSV_METRIC_ID_MODEL_CERTAINTY_DESC,
		.version = 1,
		.num_rows = TEST_ROWS,
		.num_cols = TEST_COLS,
		.counts = priv,
	};
}

static nrf_edgeai_obsv_metric_t metric = {
	.init = metric_init,
	.update = metric_update,
	.clear = metric_clear,
	.snapshot = metric_snapshot,
	.source = NRF_EDGEAI_OBSV_SOURCE_PROBS,
	.priv = counts,
};

#define TEST_MODEL_ID      3U
#define TEST_MODEL_VERSION 7U

enum {
	KEY_MODEL_ID = 13,
	KEY_MODEL_VERSION,
	KEY_NUM_INFERENCES,
	KEY_NUM_FEATURE_UPDATES,
	KEY_MCD_VERSION,
};

static const struct nrf_edgeai_obsv_memfault_keys mflt_keys = {
	.model = {
		.model_id = {._impl = KEY_MODEL_ID},
		.model_version = {._impl = KEY_MODEL_VERSION},
		.num_inferences = {._impl = KEY_NUM_INFERENCES},
		.num_feature_updates = {._impl = KEY_NUM_FEATURE_UPDATES},
	},
	.mcd = {
		.version = {._impl = KEY_MCD_VERSION},
		.entropy = {{._impl = 1}, {._impl = 2}, {._impl = 3}, {._impl = 4}},
		.top2_margin = {{._impl = 5}, {._impl = 6}, {._impl = 7}, {._impl = 8}},
		.switches = {._impl = 9},
		.comparisons = {._impl = 10},
		.majority_frames = {._impl = 11},
		.confident_switches = {._impl = 12},
	},
};

static nrf_edgeai_obsv_ctx_t ctx;
static const float probs[] = {0.75f, 0.25f};

static void *suite_setup(void)
{
	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = 2,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_init(&ctx, &model));
	zassert_ok(nrf_edgeai_obsv_register(&ctx, &metric, NULL));
	zassert_ok(nrf_edgeai_obsv_memfault_init(&ctx, &mflt_keys));

	return NULL;
}

static void before_each(void *fixture)
{
	ARG_UNUSED(fixture);

	memfault_mock_reset();
	zassert_ok(nrf_edgeai_obsv_reset(&ctx));
}

ZTEST(obsv_memfault, test_heartbeat_exports_bound_cells_and_resets_context)
{
	zassert_ok(nrf_edgeai_obsv_update_probs(&ctx, probs));
	zassert_ok(nrf_edgeai_obsv_update_probs(&ctx, probs));
	zassert_ok(nrf_edgeai_obsv_memfault_collect());

	zassert_equal(memfault_mock_value(1), 2U);
	zassert_equal(memfault_mock_value(7), 14U);
	zassert_equal(memfault_mock_write_count(), 17U);
	zassert_equal(ctx.state.num_inferences, 0U);

	zassert_ok(nrf_edgeai_obsv_memfault_collect());
	zassert_equal(memfault_mock_write_count(), 17U,
		      "empty interval unexpectedly wrote heartbeat metrics");
}

ZTEST(obsv_memfault, test_heartbeat_exports_model_identity_and_counters)
{
	const float feats[] = {0.0f};

	zassert_ok(nrf_edgeai_obsv_update_features(&ctx, feats, ARRAY_SIZE(feats)));
	zassert_ok(nrf_edgeai_obsv_update_probs(&ctx, probs));
	zassert_ok(nrf_edgeai_obsv_update_probs(&ctx, probs));
	zassert_ok(nrf_edgeai_obsv_memfault_collect());

	zassert_equal(memfault_mock_value(KEY_MODEL_ID), TEST_MODEL_ID);
	zassert_equal(memfault_mock_value(KEY_MODEL_VERSION), TEST_MODEL_VERSION);
	zassert_equal(memfault_mock_value(KEY_NUM_INFERENCES), 2U);
	zassert_equal(memfault_mock_value(KEY_NUM_FEATURE_UPDATES), 1U);
	zassert_equal(memfault_mock_value(KEY_MCD_VERSION), 1U);
}

ZTEST(obsv_memfault, test_heartbeat_holds_counters_when_storage_full)
{
	zassert_ok(nrf_edgeai_obsv_update_probs(&ctx, probs));

	memfault_mock_set_storage_full(true);
	zassert_equal(nrf_edgeai_obsv_memfault_collect(), -ENOSPC);
	zassert_equal(memfault_mock_write_count(), 0U, "full storage wrote heartbeat metrics");
	zassert_equal(ctx.state.num_inferences, 1U, "full storage reset the context");

	zassert_ok(nrf_edgeai_obsv_update_probs(&ctx, probs));

	memfault_mock_set_storage_full(false);
	zassert_ok(nrf_edgeai_obsv_memfault_collect());
	zassert_equal(memfault_mock_value(1), 2U, "held-back interval was not carried over");
	zassert_equal(memfault_mock_value(7), 14U);
	zassert_equal(memfault_mock_value(KEY_NUM_INFERENCES), 2U);
	zassert_equal(ctx.state.num_inferences, 0U);
}

ZTEST_SUITE(obsv_memfault, NULL, suite_setup, before_each, NULL, NULL);
