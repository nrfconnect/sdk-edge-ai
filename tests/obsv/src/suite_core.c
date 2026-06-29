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

/* Per-file metric instances — each test that needs a metric uses these.
 * Using distinct instances per test suite file means no cross-suite state.
 */
static uint32_t core_pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(TEST_NUM_CLASSES) /
			     sizeof(uint32_t)];
static uint32_t core_tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(TEST_NUM_CLASSES) /
			     sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t core_pd;
static nrf_edgeai_obsv_metric_t core_tm;

static uint32_t iso_pd_a_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(TEST_NUM_CLASSES) /
			      sizeof(uint32_t)];
static uint32_t iso_pd_b_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(TEST_NUM_CLASSES) /
			      sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t iso_pd_a;
static nrf_edgeai_obsv_metric_t iso_pd_b;

static void core_before(void *fixture)
{
	ARG_UNUSED(fixture);
	memset(&ctx, 0, sizeof(ctx));
	nrf_edgeai_obsv_core_init(&ctx, &test_model);
	nrf_edgeai_obsv_metric_pd_create(&core_pd, core_pd_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_tm_create(&core_tm, core_tm_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_pd_create(&iso_pd_a, iso_pd_a_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_pd_create(&iso_pd_b, iso_pd_b_buf, TEST_NUM_CLASSES);
}

ZTEST_SUITE(obsv_core, NULL, NULL, core_before, NULL, NULL);

ZTEST(obsv_core, test_init_copies_model)
{
	zassert_equal(ctx.model.model_id, TEST_MODEL_ID);
	zassert_equal(ctx.model.num_classes, TEST_NUM_CLASSES);
	zassert_equal(ctx.model.version, TEST_MODEL_VERSION);
}

ZTEST(obsv_core, test_init_zeroes_state)
{
	zassert_is_null(ctx.p_metrics_list);
	zassert_equal(ctx.num_inferences, 0);
	zassert_equal(ctx.num_features, 0);
	zassert_equal(ctx.num_metrics, 0);
}

ZTEST(obsv_core, test_init_null_ctx)
{
	int rc = nrf_edgeai_obsv_core_init(NULL, &test_model);

	zassert_true(rc < 0, "expected error on NULL ctx");
}

ZTEST(obsv_core, test_init_null_model)
{
	nrf_edgeai_obsv_core_t local;
	int rc = nrf_edgeai_obsv_core_init(&local, NULL);

	zassert_true(rc < 0, "expected error on NULL model");
}

ZTEST(obsv_core, test_init_num_classes_exceeds_max)
{
	nrf_edgeai_obsv_core_t local;
	const nrf_edgeai_obsv_model_info_t bad_model = {
		.model_id = 1,
		.num_classes = CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES + 1,
		.version = 1,
	};
	int rc = nrf_edgeai_obsv_core_init(&local, &bad_model);

	zassert_equal(rc, -EINVAL, "expected -EINVAL when num_classes exceeds max");
}

ZTEST(obsv_core, test_register_single_metric)
{
	int rc = nrf_edgeai_obsv_core_register(&ctx, &core_pd, NULL);

	zassert_equal(rc, 0);
	zassert_equal(ctx.num_metrics, 1);
	zassert_equal_ptr(ctx.p_metrics_list, &core_pd);
}

ZTEST(obsv_core, test_register_two_metrics)
{
	nrf_edgeai_obsv_core_register(&ctx, &core_pd, NULL);
	nrf_edgeai_obsv_core_register(&ctx, &core_tm, NULL);

	zassert_equal(ctx.num_metrics, 2);
	zassert_equal_ptr(ctx.p_metrics_list, &core_pd);
	zassert_equal_ptr(core_pd.p_next, &core_tm);
	zassert_is_null(core_tm.p_next);
}

ZTEST(obsv_core, test_register_null_params)
{
	zassert_true(nrf_edgeai_obsv_core_register(NULL, &core_pd, NULL) < 0);
	zassert_true(nrf_edgeai_obsv_core_register(&ctx, NULL, NULL) < 0);
}

ZTEST(obsv_core, test_deregister_head)
{
	nrf_edgeai_obsv_core_register(&ctx, &core_pd, NULL);
	nrf_edgeai_obsv_core_register(&ctx, &core_tm, NULL);
	nrf_edgeai_obsv_core_deregister(&ctx, &core_pd);

	zassert_equal(ctx.num_metrics, 1);
	zassert_equal_ptr(ctx.p_metrics_list, &core_tm);
	zassert_is_null(core_pd.p_next);
}

ZTEST(obsv_core, test_deregister_tail)
{
	nrf_edgeai_obsv_core_register(&ctx, &core_pd, NULL);
	nrf_edgeai_obsv_core_register(&ctx, &core_tm, NULL);
	nrf_edgeai_obsv_core_deregister(&ctx, &core_tm);

	zassert_equal(ctx.num_metrics, 1);
	zassert_equal_ptr(ctx.p_metrics_list, &core_pd);
	zassert_is_null(core_pd.p_next);
}

ZTEST(obsv_core, test_deregister_absent_is_noop)
{
	nrf_edgeai_obsv_core_register(&ctx, &core_pd, NULL);
	nrf_edgeai_obsv_core_deregister(&ctx, &core_tm);

	zassert_equal(ctx.num_metrics, 1);
	zassert_equal_ptr(ctx.p_metrics_list, &core_pd);
}

ZTEST(obsv_core, test_update_increments_counter)
{
	nrf_edgeai_obsv_core_register(&ctx, &core_pd, NULL);

	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	nrf_edgeai_obsv_core_update_probs(&ctx, probs);
	nrf_edgeai_obsv_core_update_probs(&ctx, probs);
	nrf_edgeai_obsv_core_update_probs(&ctx, probs);

	zassert_equal(ctx.num_inferences, 3);
}

ZTEST(obsv_core, test_update_features_increments_counter)
{
	const float feats[4] = {0.1f, 0.2f, 0.3f, 0.4f};
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	nrf_edgeai_obsv_core_update_features(&ctx, feats, ARRAY_SIZE(feats));
	nrf_edgeai_obsv_core_update_features(&ctx, feats, ARRAY_SIZE(feats));

	zassert_equal(ctx.num_features, 2U);
	zassert_equal(ctx.num_inferences, 0U, "feature updates must not count inferences");

	/* The two stream counters are independent. */
	nrf_edgeai_obsv_core_update_probs(&ctx, probs);

	zassert_equal(ctx.num_inferences, 1U);
	zassert_equal(ctx.num_features, 2U, "inference updates must not count features");
}

ZTEST(obsv_core, test_update_features_null_params)
{
	const float feats[4] = {0.1f, 0.2f, 0.3f, 0.4f};

	zassert_true(nrf_edgeai_obsv_core_update_features(NULL, feats, ARRAY_SIZE(feats)) < 0);
	zassert_true(nrf_edgeai_obsv_core_update_features(&ctx, NULL, ARRAY_SIZE(feats)) < 0);
}

ZTEST(obsv_core, test_update_null_params)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	zassert_true(nrf_edgeai_obsv_core_update_probs(NULL, probs) < 0);
	zassert_true(nrf_edgeai_obsv_core_update_probs(&ctx, NULL) < 0);
}

ZTEST(obsv_core, test_reset_clears_inferences)
{
	nrf_edgeai_obsv_core_register(&ctx, &core_pd, NULL);

	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};
	const float feats[4] = {0.1f, 0.2f, 0.3f, 0.4f};

	nrf_edgeai_obsv_core_update_probs(&ctx, probs);
	nrf_edgeai_obsv_core_update_features(&ctx, feats, ARRAY_SIZE(feats));
	nrf_edgeai_obsv_core_reset(&ctx);

	zassert_equal(ctx.num_inferences, 0);
	zassert_equal(ctx.num_features, 0);
	zassert_equal(ctx.num_metrics, 1, "reset must keep metrics");
	zassert_equal(ctx.model.model_id, TEST_MODEL_ID, "reset must keep model");
}

/*
 * Two-context isolation test: feed inferences into model A only and verify
 * that model B's metric state remains zero. Each context uses its own metric
 * instance (no shared static storage).
 */
ZTEST(obsv_core, test_two_contexts_no_shared_state)
{
	nrf_edgeai_obsv_core_t ctx_a;
	nrf_edgeai_obsv_core_t ctx_b;

	const nrf_edgeai_obsv_model_info_t model_a = {
		.model_id = 10U,
		.num_classes = TEST_NUM_CLASSES,
		.version = 1U,
	};
	const nrf_edgeai_obsv_model_info_t model_b = {
		.model_id = 20U,
		.num_classes = TEST_NUM_CLASSES,
		.version = 1U,
	};

	zassert_equal(nrf_edgeai_obsv_core_init(&ctx_a, &model_a), 0);
	zassert_equal(nrf_edgeai_obsv_core_init(&ctx_b, &model_b), 0);

	zassert_equal(nrf_edgeai_obsv_core_register(&ctx_a, &iso_pd_a, NULL), 0);
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx_b, &iso_pd_b, NULL), 0);

	/* Feed 2 inferences exclusively to model A. */
	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};

	nrf_edgeai_obsv_core_update_probs(&ctx_a, probs);
	nrf_edgeai_obsv_core_update_probs(&ctx_a, probs);

	zassert_equal(ctx_a.num_inferences, 2U);
	zassert_equal(ctx_b.num_inferences, 0U);

	/* Snapshot both contexts. */
	struct test_snapshots snaps_a = {0};
	struct test_snapshots snaps_b = {0};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx_a, test_capture_cb, &snaps_a), 0);
	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx_b, test_capture_cb, &snaps_b), 0);

	/* Sum all histogram entries for each model. */
	uint32_t sum_a = 0;
	uint32_t sum_b = 0;
	uint32_t total_cells =
		(uint32_t)snaps_a.probs_distribution.num_rows * snaps_a.probs_distribution.num_cols;

	for (uint32_t i = 0; i < total_cells; i++) {
		sum_a += snaps_a.probs_distribution.counts[i];
		sum_b += snaps_b.probs_distribution.counts[i];
	}

	zassert_equal(sum_a, 2U * TEST_NUM_CLASSES,
		      "model A must have 2*num_classes histogram entries");
	zassert_equal(sum_b, 0U, "model B must have 0 entries — instances are independent");
}
