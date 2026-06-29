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

static uint32_t payload_pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(TEST_NUM_CLASSES) /
			       sizeof(uint32_t)];
static uint32_t payload_tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(TEST_NUM_CLASSES) /
			       sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t payload_pd;
static nrf_edgeai_obsv_metric_t payload_tm;

static void payload_before(void *fixture)
{
	ARG_UNUSED(fixture);
	memset(&ctx, 0, sizeof(ctx));
	nrf_edgeai_obsv_core_init(&ctx, &test_model);
	nrf_edgeai_obsv_metric_pd_create(&payload_pd, payload_pd_buf, TEST_NUM_CLASSES);
	nrf_edgeai_obsv_metric_tm_create(&payload_tm, payload_tm_buf, TEST_NUM_CLASSES);
}

ZTEST_SUITE(obsv_payload, NULL, NULL, payload_before, NULL, NULL);

/* Translate the flat row-major counts from the snapshot back to a
 * [class][bin] cell for readable assertions.
 */
static uint32_t probs_cell(const struct test_metric_capture *cap, uint16_t c, uint16_t b)
{
	return cap->counts[c * cap->num_cols + b];
}

static uint32_t tm_cell(const struct test_metric_capture *cap, uint16_t i, uint16_t j)
{
	return cap->counts[i * cap->num_cols + j];
}

static void capture_snapshots(struct test_snapshots *out)
{
	memset(out, 0, sizeof(*out));
	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, out), 0);
}

ZTEST(obsv_payload, test_probs_distribution_default_bins)
{
	nrf_edgeai_obsv_core_register(&ctx, &payload_pd, NULL);

	/* 4 uniform bins over [0,1]: edges {0, 0.25, 0.5, 0.75, 1.0} */
	const float p1[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};
	const float p2[TEST_NUM_CLASSES] = {0.1f, 0.2f, 0.6f, 0.1f};
	const float p3[TEST_NUM_CLASSES] = {0.0f, 0.2f, 0.8f, 0.0f};

	nrf_edgeai_obsv_core_update_probs(&ctx, p1);
	nrf_edgeai_obsv_core_update_probs(&ctx, p2);
	nrf_edgeai_obsv_core_update_probs(&ctx, p3);

	struct test_snapshots snaps;

	capture_snapshots(&snaps);
	zassert_true(snaps.probs_distribution.present);

	const struct test_metric_capture *pd = &snaps.probs_distribution;

	/* class 0: 1.0→bin3, 0.1→bin0, 0.0→bin0 */
	zassert_equal(probs_cell(pd, 0, 0), 2);
	zassert_equal(probs_cell(pd, 0, 3), 1);

	/* class 1: 0.0→bin0, 0.2→bin0, 0.2→bin0 */
	zassert_equal(probs_cell(pd, 1, 0), 3);

	/* class 2: 0.0→bin0, 0.6→bin2, 0.8→bin3 */
	zassert_equal(probs_cell(pd, 2, 0), 1);
	zassert_equal(probs_cell(pd, 2, 2), 1);
	zassert_equal(probs_cell(pd, 2, 3), 1);
}

ZTEST(obsv_payload, test_probs_distribution_custom_edges)
{
	/* 4 bins with custom non-uniform edges (inner edges only; 0.0 and 1.0 are implicit) */
	const nrf_obsv_probs_dist_cfg_t cfg = {
		.bin_edges = {0.1f, 0.5f, 0.9f},
	};

	nrf_edgeai_obsv_core_register(&ctx, &payload_pd, &cfg);

	const float probs[TEST_NUM_CLASSES] = {0.05f, 0.3f, 0.95f, 1.0f};

	nrf_edgeai_obsv_core_update_probs(&ctx, probs);

	struct test_snapshots snaps;

	capture_snapshots(&snaps);
	zassert_true(snaps.probs_distribution.present);

	const struct test_metric_capture *pd = &snaps.probs_distribution;

	/* 0.05 → bin 0 ([0.0, 0.1)) */
	zassert_equal(probs_cell(pd, 0, 0), 1);
	/* 0.3 → bin 1 ([0.1, 0.5)) */
	zassert_equal(probs_cell(pd, 1, 1), 1);
	/* 0.95 → bin 3 ([0.9, 1.0]) */
	zassert_equal(probs_cell(pd, 2, 3), 1);
	/* 1.0 → bin 3 (clamped to last) */
	zassert_equal(probs_cell(pd, 3, 3), 1);
}

ZTEST(obsv_payload, test_transition_matrix_values)
{
	nrf_edgeai_obsv_core_register(&ctx, &payload_tm, NULL);

	/* argmax sequence: 0 → 2 → 2 */
	const float p1[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};
	const float p2[TEST_NUM_CLASSES] = {0.1f, 0.2f, 0.6f, 0.1f};
	const float p3[TEST_NUM_CLASSES] = {0.0f, 0.2f, 0.8f, 0.0f};

	nrf_edgeai_obsv_core_update_probs(&ctx, p1);
	nrf_edgeai_obsv_core_update_probs(&ctx, p2);
	nrf_edgeai_obsv_core_update_probs(&ctx, p3);

	struct test_snapshots snaps;

	capture_snapshots(&snaps);
	zassert_true(snaps.transition_matrix.present);

	const struct test_metric_capture *tm = &snaps.transition_matrix;

	zassert_equal(tm_cell(tm, 0, 2), 1, "transition 0→2");
	zassert_equal(tm_cell(tm, 2, 2), 1, "transition 2→2");
	zassert_equal(tm_cell(tm, 1, 1), 0, "no 1→1 transition");
}

ZTEST(obsv_payload, test_reset_clears_metric_state)
{
	nrf_edgeai_obsv_core_register(&ctx, &payload_pd, NULL);

	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};

	nrf_edgeai_obsv_core_update_probs(&ctx, probs);
	nrf_edgeai_obsv_core_reset(&ctx);

	struct test_snapshots snaps;

	capture_snapshots(&snaps);
	zassert_true(snaps.probs_distribution.present);

	const struct test_metric_capture *pd = &snaps.probs_distribution;
	uint32_t sum = 0;

	for (uint16_t c = 0; c < pd->num_rows; c++) {
		for (uint16_t b = 0; b < pd->num_cols; b++) {
			sum += probs_cell(pd, c, b);
		}
	}

	zassert_equal(sum, 0, "all bins must be zero after reset");
}

ZTEST(obsv_payload, test_reset_preserves_custom_bin_edges)
{
	/* Non-uniform edges: bin 0 = [0.0, 0.5), bin 1 = [0.5, 0.9), ... (0.0 and 1.0 implicit) */
	const nrf_obsv_probs_dist_cfg_t cfg = {
		.bin_edges = {0.5f, 0.9f, 0.95f},
	};

	nrf_edgeai_obsv_core_register(&ctx, &payload_pd, &cfg);

	const float before[TEST_NUM_CLASSES] = {0.3f, 0.0f, 0.0f, 0.0f};

	nrf_edgeai_obsv_core_update_probs(&ctx, before);
	nrf_edgeai_obsv_core_reset(&ctx);

	/* 0.3 must still land in bin 0 ([0.0, 0.5)) after reset, not bin 1
	 * ([0.25, 0.5) with default uniform edges) — proves edges were kept.
	 */
	const float after[TEST_NUM_CLASSES] = {0.3f, 0.0f, 0.0f, 0.0f};

	nrf_edgeai_obsv_core_update_probs(&ctx, after);

	struct test_snapshots snaps;

	capture_snapshots(&snaps);
	zassert_true(snaps.probs_distribution.present);

	const struct test_metric_capture *pd = &snaps.probs_distribution;

	zassert_equal(probs_cell(pd, 0, 0), 1,
		      "0.3 must be in bin 0 after reset (custom edges preserved)");
	zassert_equal(probs_cell(pd, 0, 1), 0, "bin 1 must be empty");
}
