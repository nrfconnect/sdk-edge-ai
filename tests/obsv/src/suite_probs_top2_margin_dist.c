/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define MARGIN_BINS CONFIG_NRF_EDGEAI_OBSV_PROBS_TOP2_MARGIN_DIST_BIN_NUM

struct pmd_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t bins[16];
};

static bool pmd_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct pmd_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_PROBS_TOP2_MARGIN_DIST) {
		return true;
	}

	cap->present = true;
	cap->metric_id = snap->metric_id;
	cap->version = snap->version;
	cap->num_rows = snap->num_rows;
	cap->num_cols = snap->num_cols;
	for (size_t i = 0; i < (size_t)snap->num_cols && i < ARRAY_SIZE(cap->bins); i++) {
		cap->bins[i] = snap->counts[i];
	}

	return true;
}

static nrf_edgeai_obsv_core_t ctx;
static uint32_t pmd_buf[NRF_EDGEAI_OBSV_PMD_STORAGE_BYTES(TEST_NUM_CLASSES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t pmd_metric;

static void pmd_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_pmd_create(&pmd_metric, pmd_buf, TEST_NUM_CLASSES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &pmd_metric, NULL));
}

/* Feed one probability vector of TEST_NUM_CLASSES entries. */
static void feed(const float *probs)
{
	zassert_ok(nrf_edgeai_obsv_core_update_probs(&ctx, probs));
}

static struct pmd_capture capture(void)
{
	struct pmd_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, pmd_capture_cb, &cap));
	zassert_true(cap.present, "margin metric snapshot not visited");

	return cap;
}

static uint32_t total(const struct pmd_capture *cap)
{
	uint32_t sum = 0;

	for (uint16_t i = 0; i < cap->num_cols; i++) {
		sum += cap->bins[i];
	}
	return sum;
}

ZTEST_SUITE(obsv_pmd, NULL, NULL, pmd_setup, NULL, NULL);

/* 1 x bin_num histogram, id 6, version 1. TEST_NUM_CLASSES == 4 == MARGIN_BINS here. */
ZTEST(obsv_pmd, test_snapshot_shape)
{
	struct pmd_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_PROBS_TOP2_MARGIN_DIST);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.num_rows, 1);
	zassert_equal(cap.num_cols, MARGIN_BINS);
}

/* One-hot vector -> margin = 1 - 0 = 1 -> top bin (most decisive). */
ZTEST(obsv_pmd, test_one_hot_is_max_margin)
{
	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};

	feed(probs);

	struct pmd_capture cap = capture();

	zassert_equal(cap.bins[MARGIN_BINS - 1], 1, "max margin must land in the top bin");
	zassert_equal(total(&cap), 1);
}

/* Uniform vector -> top1 == top2 -> margin = 0 -> bin 0 (fully ambiguous). */
ZTEST(obsv_pmd, test_uniform_is_min_margin)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	feed(probs);

	struct pmd_capture cap = capture();

	zassert_equal(cap.bins[0], 1, "zero margin must land in bin 0");
	zassert_equal(total(&cap), 1);
}

/*
 * Two near-tied top classes -> tiny margin even though no class dominates.
 * margin([0.41, 0.39, 0.1, 0.1]) = 0.41 - 0.39 = 0.02, which falls in
 * [0, 0.25) -> bin 0.
 */
ZTEST(obsv_pmd, test_ambiguous_is_low_margin)
{
	const float probs[TEST_NUM_CLASSES] = {0.41f, 0.39f, 0.1f, 0.1f};

	feed(probs);

	struct pmd_capture cap = capture();

	zassert_equal(cap.bins[0], 1, "near-tied top-2 must land in bin 0");
	zassert_equal(total(&cap), 1);
}

/*
 * A margin clear of the bin edges lands in a middle bin.
 * margin([0.8, 0.2, 0.0, 0.0]) = 0.6, which falls in [0.5, 0.75) -> bin 2 of 4.
 */
ZTEST(obsv_pmd, test_mid_margin_lands_in_middle_bin)
{
	const float probs[TEST_NUM_CLASSES] = {0.8f, 0.2f, 0.0f, 0.0f};

	feed(probs);

	struct pmd_capture cap = capture();

	zassert_equal(cap.bins[2], 1, "margin 0.6 must land in bin 2");
	zassert_equal(total(&cap), 1);
}

/* Every update increments exactly one bin. */
ZTEST(obsv_pmd, test_total_equals_update_count)
{
	const float a[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};
	const float b[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};
	const float c[TEST_NUM_CLASSES] = {0.8f, 0.2f, 0.0f, 0.0f};

	feed(a);
	feed(b);
	feed(c);

	struct pmd_capture cap = capture();

	zassert_equal(total(&cap), 3);
}

/* reset() zeroes the histogram. */
ZTEST(obsv_pmd, test_reset_clears_histogram)
{
	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};

	feed(probs);
	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));

	struct pmd_capture cap = capture();

	zassert_equal(total(&cap), 0);
}
