/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define MSD_BINS     CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM
#define MSD_ROWS     NRF_EDGEAI_OBSV_MSD_NUM_ROWS
/* 6 features split into clean thirds: low={0,1} mid={2,3} high={4,5}. */
#define MSD_FEATURES 6

/* Row layout, mirrors the metric implementation. */
#define ROW_LOW      0
#define ROW_MID      1
#define ROW_HIGH     2
#define ROW_CENTROID 3
#define ROW_SPREAD   4
#define ROW_ENTROPY  5
#define ROW_FLATNESS 6
#define ROW_CONTRAST 7

struct msd_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t counts[MSD_ROWS * 16];
};

static bool msd_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct msd_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_MEL_SPECTRAL_DESC) {
		return true;
	}

	cap->present = true;
	cap->metric_id = snap->metric_id;
	cap->version = snap->version;
	cap->num_rows = snap->num_rows;
	cap->num_cols = snap->num_cols;

	uint32_t n = (uint32_t)snap->num_rows * snap->num_cols;

	for (uint32_t i = 0; i < n && i < ARRAY_SIZE(cap->counts); i++) {
		cap->counts[i] = snap->counts[i];
	}

	return true;
}

static nrf_edgeai_obsv_core_t ctx;
static uint32_t msd_buf[NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(MSD_FEATURES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t msd_metric;

static void msd_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_msd_create(&msd_metric, msd_buf, MSD_FEATURES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &msd_metric, NULL));
}

/* Feed one feature vector of @n entries on the FEATURES stream. */
static void feed(const float *feats, uint16_t n)
{
	zassert_ok(nrf_edgeai_obsv_core_update_features(&ctx, feats, n));
}

static struct msd_capture capture(void)
{
	struct msd_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, msd_capture_cb, &cap));
	zassert_true(cap.present, "mel spectral descriptor snapshot not visited");

	return cap;
}

/* Bin counter at [row, col]. */
static uint32_t bin(const struct msd_capture *cap, uint16_t row, uint16_t col)
{
	return cap->counts[row * cap->num_cols + col];
}

/* Sum of one row's bins. */
static uint32_t row_total(const struct msd_capture *cap, uint16_t row)
{
	uint32_t sum = 0;

	for (uint16_t c = 0; c < cap->num_cols; c++) {
		sum += bin(cap, row, c);
	}
	return sum;
}

ZTEST_SUITE(obsv_msd, NULL, NULL, msd_setup, NULL, NULL);

/* 8 x bin_num descriptor, id 8, version 1. */
ZTEST(obsv_msd, test_snapshot_shape)
{
	struct msd_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_MEL_SPECTRAL_DESC);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.num_rows, MSD_ROWS);
	zassert_equal(cap.num_cols, MSD_BINS);
}

/*
 * Flat spectrum (all bins equal). Energy splits evenly across the three bands
 * (each 1/3 -> bin 1). Entropy is maximal -> top bin; flatness is ~1 -> top bin;
 * contrast (max == min) is 0 -> bin 0. Centroid sits exactly on a bin edge
 * (0.5), so it is intentionally not asserted here.
 */
ZTEST(obsv_msd, test_flat_spectrum)
{
	const float feats[MSD_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(feats, MSD_FEATURES);

	struct msd_capture cap = capture();

	zassert_equal(bin(&cap, ROW_LOW, 1), 1, "1/3 band ratio -> bin 1");
	zassert_equal(bin(&cap, ROW_MID, 1), 1);
	zassert_equal(bin(&cap, ROW_HIGH, 1), 1);
	zassert_equal(bin(&cap, ROW_ENTROPY, MSD_BINS - 1), 1, "uniform spectrum -> max entropy");
	zassert_equal(bin(&cap, ROW_FLATNESS, MSD_BINS - 1), 1, "flat spectrum -> flatness ~1");
	zassert_equal(bin(&cap, ROW_CONTRAST, 0), 1, "no contrast -> bin 0");
}

/*
 * All energy in the single top bin. The high band gets the whole ratio (1 -> top
 * bin), low/mid get none (bin 0). The centroid is pinned at the last index
 * (-> 1.0 -> top bin) with zero spread (bin 0); entropy is 0 (bin 0); the
 * spectrum is maximally peaky so flatness -> bin 0 and contrast -> top bin.
 */
ZTEST(obsv_msd, test_concentrated_high_band)
{
	const float feats[MSD_FEATURES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f};

	feed(feats, MSD_FEATURES);

	struct msd_capture cap = capture();

	zassert_equal(bin(&cap, ROW_LOW, 0), 1);
	zassert_equal(bin(&cap, ROW_MID, 0), 1);
	zassert_equal(bin(&cap, ROW_HIGH, MSD_BINS - 1), 1);
	zassert_equal(bin(&cap, ROW_CENTROID, MSD_BINS - 1), 1);
	zassert_equal(bin(&cap, ROW_SPREAD, 0), 1);
	zassert_equal(bin(&cap, ROW_ENTROPY, 0), 1);
	zassert_equal(bin(&cap, ROW_FLATNESS, 0), 1);
	zassert_equal(bin(&cap, ROW_CONTRAST, MSD_BINS - 1), 1);
}

/*
 * Negative feature values are clamped to 0 before the spectral math, so a frame
 * with a negative entry must produce exactly the same histogram as the same
 * frame with that entry zeroed. This guards logf()/sqrtf() and the p[b]=x/S
 * probabilities against NaN.
 */
ZTEST(obsv_msd, test_negative_inputs_clamped)
{
	const float with_neg[MSD_FEATURES] = {-5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	const float clamped[MSD_FEATURES] = {0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(with_neg, MSD_FEATURES);
	struct msd_capture neg = capture();

	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));

	feed(clamped, MSD_FEATURES);
	struct msd_capture pos = capture();

	for (uint16_t r = 0; r < MSD_ROWS; r++) {
		for (uint16_t c = 0; c < MSD_BINS; c++) {
			zassert_equal(bin(&neg, r, c), bin(&pos, r, c),
				      "negative input must bin like its clamped value [r=%u c=%u]",
				      r, c);
		}
		/* And every row still received exactly one increment (no NaN drop). */
		zassert_equal(row_total(&neg, r), 1);
	}
}

/* Each update increments exactly one bin per row. */
ZTEST(obsv_msd, test_one_increment_per_row_per_update)
{
	const float a[MSD_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	const float b[MSD_FEATURES] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f};

	feed(a, MSD_FEATURES);
	feed(b, MSD_FEATURES);

	struct msd_capture cap = capture();

	for (uint16_t r = 0; r < MSD_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 2, "row %u must total the update count", r);
	}
}

/* Empty feature vector (n == 0) is a no-op. */
ZTEST(obsv_msd, test_empty_vector_is_noop)
{
	const float feats[MSD_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(feats, 0);

	struct msd_capture cap = capture();

	for (uint16_t r = 0; r < MSD_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 0);
	}
}

/* reset() zeroes every row. */
ZTEST(obsv_msd, test_reset_clears_histogram)
{
	const float feats[MSD_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(feats, MSD_FEATURES);
	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));

	struct msd_capture cap = capture();

	for (uint16_t r = 0; r < MSD_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 0);
	}
}

/* The metric consumes FEATURES; a probability update must not touch it. */
ZTEST(obsv_msd, test_ignores_probs_stream)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	zassert_ok(nrf_edgeai_obsv_core_update_probs(&ctx, probs));

	struct msd_capture cap = capture();

	for (uint16_t r = 0; r < MSD_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 0,
			"FEATURES metric must ignore the PROBS stream");
	}
}
