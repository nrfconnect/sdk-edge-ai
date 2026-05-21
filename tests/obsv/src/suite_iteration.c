/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

static nrf_edgeai_obsv_core_t ctx;

static const nrf_edgeai_obsv_model_info_t test_model = {
	.model_id = TEST_MODEL_ID,
	.num_classes = TEST_NUM_CLASSES,
	.version = TEST_MODEL_VERSION,
};

static uint32_t iter_pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(TEST_NUM_CLASSES) /
			     sizeof(uint32_t)];
static uint32_t iter_tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(TEST_NUM_CLASSES) /
			     sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t iter_pd;
static nrf_edgeai_obsv_metric_t iter_tm;

static void iter_before(void *fixture)
{
	ARG_UNUSED(fixture);
	memset(&ctx, 0, sizeof(ctx));
	nrf_edgeai_obsv_core_init(&ctx, &test_model);
	nrf_edgeai_obsv_metric_pd_create(&iter_pd, iter_pd_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_tm_create(&iter_tm, iter_tm_buf, TEST_NUM_CLASSES);
}

ZTEST_SUITE(obsv_iteration, NULL, NULL, iter_before, NULL, NULL);

ZTEST(obsv_iteration, test_iterate_no_metrics)
{
	struct test_snapshots snaps = {0};

	int rc = nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps);

	zassert_equal(rc, 0);
	zassert_equal(snaps.visited, 0);
	zassert_false(snaps.probs_distribution.present);
	zassert_false(snaps.transition_matrix.present);
}

ZTEST(obsv_iteration, test_iterate_two_metrics_visits_both)
{
	nrf_edgeai_obsv_core_register(&ctx, &iter_pd, NULL);
	nrf_edgeai_obsv_core_register(&ctx, &iter_tm, NULL);

	struct test_snapshots snaps = {0};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps), 0);

	zassert_equal(snaps.visited, 2);
	zassert_true(snaps.probs_distribution.present);
	zassert_true(snaps.transition_matrix.present);

	zassert_equal(snaps.probs_distribution.metric_id,
		      NRF_EDGEAI_OBSV_METRIC_ID_PROBS_DISTRIBUTION);
	zassert_equal(snaps.probs_distribution.num_rows, TEST_NUM_CLASSES);
	zassert_equal(snaps.probs_distribution.num_cols, TEST_NUM_BINS);

	zassert_equal(snaps.transition_matrix.metric_id,
		      NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX);
	zassert_equal(snaps.transition_matrix.num_rows, TEST_NUM_CLASSES);
	zassert_equal(snaps.transition_matrix.num_cols, TEST_NUM_CLASSES);
}

ZTEST(obsv_iteration, test_null_params)
{
	struct test_snapshots snaps = {0};

	zassert_true(nrf_edgeai_obsv_core_for_each_metric(NULL, test_capture_cb, &snaps) < 0);
	zassert_true(nrf_edgeai_obsv_core_for_each_metric(&ctx, NULL, &snaps) < 0);
}

static bool stop_after_first(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	uint32_t *count = user;

	ARG_UNUSED(snap);
	(*count)++;
	return false;
}

ZTEST(obsv_iteration, test_callback_can_stop_early)
{
	nrf_edgeai_obsv_core_register(&ctx, &iter_pd, NULL);
	nrf_edgeai_obsv_core_register(&ctx, &iter_tm, NULL);

	uint32_t visited = 0;

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, stop_after_first, &visited), 0);
	zassert_equal(visited, 1, "callback returning false must halt iteration");
}

ZTEST(obsv_iteration, test_after_deregister)
{
	nrf_edgeai_obsv_core_register(&ctx, &iter_pd, NULL);
	nrf_edgeai_obsv_core_register(&ctx, &iter_tm, NULL);
	nrf_edgeai_obsv_core_deregister(&ctx, &iter_tm);

	struct test_snapshots snaps = {0};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps), 0);

	zassert_equal(snaps.visited, 1);
	zassert_true(snaps.probs_distribution.present);
	zassert_false(snaps.transition_matrix.present);
}
