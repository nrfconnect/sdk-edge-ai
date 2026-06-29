/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define ENT_BINS CONFIG_NRF_EDGEAI_OBSV_PROBS_ENTROPY_DIST_BIN_NUM

struct ped_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t bins[16];
};

static bool ped_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct ped_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_PROBS_ENTROPY_DIST) {
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
static uint32_t ped_buf[NRF_EDGEAI_OBSV_PED_STORAGE_BYTES(TEST_NUM_CLASSES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t ped_metric;

static void ped_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_ped_create(&ped_metric, ped_buf, TEST_NUM_CLASSES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &ped_metric, NULL));
}

/* Feed one probability vector of TEST_NUM_CLASSES entries. */
static void feed(const float *probs)
{
	zassert_ok(nrf_edgeai_obsv_core_update_probs(&ctx, probs));
}

static struct ped_capture capture(void)
{
	struct ped_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, ped_capture_cb, &cap));
	zassert_true(cap.present, "entropy metric snapshot not visited");

	return cap;
}

static uint32_t total(const struct ped_capture *cap)
{
	uint32_t sum = 0;

	for (uint16_t i = 0; i < cap->num_cols; i++) {
		sum += cap->bins[i];
	}
	return sum;
}

ZTEST_SUITE(obsv_ped, NULL, NULL, ped_setup, NULL, NULL);

/* 1 x bin_num histogram, id 5, version 1. TEST_NUM_CLASSES == 4 == ENT_BINS here. */
ZTEST(obsv_ped, test_snapshot_shape)
{
	struct ped_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_PROBS_ENTROPY_DIST);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.num_rows, 1);
	zassert_equal(cap.num_cols, ENT_BINS);
}

/* One-hot vector -> H = 0 -> normalized 0 -> lowest bin. */
ZTEST(obsv_ped, test_one_hot_is_min_entropy)
{
	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};

	feed(probs);

	struct ped_capture cap = capture();

	zassert_equal(cap.bins[0], 1, "zero entropy must land in bin 0");
	zassert_equal(total(&cap), 1);
}

/* Uniform vector -> H = ln(N) -> normalized 1 -> top bin. */
ZTEST(obsv_ped, test_uniform_is_max_entropy)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	feed(probs);

	struct ped_capture cap = capture();

	zassert_equal(cap.bins[ENT_BINS - 1], 1, "max entropy must land in the top bin");
	zassert_equal(total(&cap), 1);
}

/*
 * Skewed-but-spread vector lands in a middle bin, clear of the bin edges.
 * H([0.7,0.1,0.1,0.1]) ≈ 0.940 nats; normalized ≈ 0.940 / ln(4) ≈ 0.679,
 * which falls in [0.5, 0.75) -> bin 2 of 4 uniform bins.
 */
ZTEST(obsv_ped, test_mid_entropy_lands_in_middle_bin)
{
	const float probs[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};

	feed(probs);

	struct ped_capture cap = capture();

	zassert_equal(cap.bins[2], 1, "normalized entropy ~0.68 must land in bin 2");
	zassert_equal(total(&cap), 1);
}

/* Every update increments exactly one bin. */
ZTEST(obsv_ped, test_total_equals_update_count)
{
	const float a[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};
	const float b[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};
	const float c[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};

	feed(a);
	feed(b);
	feed(c);

	struct ped_capture cap = capture();

	zassert_equal(total(&cap), 3);
}

/* reset() zeroes the histogram. */
ZTEST(obsv_ped, test_reset_clears_histogram)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	feed(probs);
	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));

	struct ped_capture cap = capture();

	zassert_equal(total(&cap), 0);
}
