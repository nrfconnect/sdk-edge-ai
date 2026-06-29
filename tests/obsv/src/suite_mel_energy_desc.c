/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define MED_BINS     CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM
#define MED_ROWS     NRF_EDGEAI_OBSV_MED_NUM_ROWS
#define MED_FEATURES 8

/* Row layout, mirrors the metric implementation. */
#define ROW_MEAN  0
#define ROW_MAX   1
#define ROW_DYN   2
#define ROW_FLOOR 3

struct med_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t counts[MED_ROWS * 16];
};

static bool med_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct med_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_MEL_ENERGY_DESC) {
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
static uint32_t med_buf[NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(MED_FEATURES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t med_metric;

static void med_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_med_create(&med_metric, med_buf, MED_FEATURES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &med_metric, NULL));
}

/* Feed one feature vector of MED_FEATURES entries on the FEATURES stream. */
static void feed(const float *feats)
{
	zassert_ok(nrf_edgeai_obsv_core_update_features(&ctx, feats, MED_FEATURES));
}

static struct med_capture capture(void)
{
	struct med_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, med_capture_cb, &cap));
	zassert_true(cap.present, "mel energy descriptor snapshot not visited");

	return cap;
}

/* Bin counter at [row, col]. */
static uint32_t bin(const struct med_capture *cap, uint16_t row, uint16_t col)
{
	return cap->counts[row * cap->num_cols + col];
}

/* Sum of one row's bins. */
static uint32_t row_total(const struct med_capture *cap, uint16_t row)
{
	uint32_t sum = 0;

	for (uint16_t c = 0; c < cap->num_cols; c++) {
		sum += bin(cap, row, c);
	}
	return sum;
}

ZTEST_SUITE(obsv_med, NULL, NULL, med_setup, NULL, NULL);

/*
 * The prj.conf percentile scale is [p01, p99] = [0, 1.0], so x_norm == clamp(x).
 * With MED_BINS == 4 the uniform [0, 1] bins are
 * [0, .25) [.25, .5) [.5, .75) [.75, 1].
 */

/* 4 x bin_num descriptor, id 7, version 1. */
ZTEST(obsv_med, test_snapshot_shape)
{
	struct med_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_MEL_ENERGY_DESC);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.num_rows, MED_ROWS);
	zassert_equal(cap.num_cols, MED_BINS);
}

/*
 * All-zero frame: mean/max/range are 0 -> bin 0; every bin is at or below the
 * raw silence floor -> floor_ratio == 1 -> top bin.
 */
ZTEST(obsv_med, test_silence_all_zero)
{
	const float feats[MED_FEATURES] = {0};

	feed(feats);

	struct med_capture cap = capture();

	zassert_equal(bin(&cap, ROW_MEAN, 0), 1);
	zassert_equal(bin(&cap, ROW_MAX, 0), 1);
	zassert_equal(bin(&cap, ROW_DYN, 0), 1);
	zassert_equal(bin(&cap, ROW_FLOOR, MED_BINS - 1), 1,
		"floor_ratio 1.0 must land in top bin");
}

/*
 * All-ones frame: mean == max == 1 -> top bin; the frame is flat so the dynamic
 * range is 0 -> bin 0; nothing is at the silence floor -> floor_ratio 0 -> bin 0.
 */
ZTEST(obsv_med, test_full_scale_all_one)
{
	const float feats[MED_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(feats);

	struct med_capture cap = capture();

	zassert_equal(bin(&cap, ROW_MEAN, MED_BINS - 1), 1);
	zassert_equal(bin(&cap, ROW_MAX, MED_BINS - 1), 1);
	zassert_equal(bin(&cap, ROW_DYN, 0), 1);
	zassert_equal(bin(&cap, ROW_FLOOR, 0), 1);
}

/*
 * Half-silent / half-full frame: mean 0.5 -> bin 2; max 1 -> top bin; the q95-q05
 * spread is the full 1.0 -> top bin; half the bins sit at the floor -> 0.5 -> bin 2.
 */
ZTEST(obsv_med, test_spread_and_floor)
{
	const float feats[MED_FEATURES] = {0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(feats);

	struct med_capture cap = capture();

	zassert_equal(bin(&cap, ROW_MEAN, 2), 1);
	zassert_equal(bin(&cap, ROW_MAX, MED_BINS - 1), 1);
	zassert_equal(bin(&cap, ROW_DYN, MED_BINS - 1), 1);
	zassert_equal(bin(&cap, ROW_FLOOR, 2), 1);
}

/*
 * Values above p99 must be clamped into [0, 1] before being averaged. The frame
 * alternates 2.0 and 0.0: if the 2.0 leaked through unclamped the mean would be
 * 1.0 (top bin); clamped to 1.0 the mean is 0.5 -> bin 2.
 */
ZTEST(obsv_med, test_clamps_above_p99)
{
	const float feats[MED_FEATURES] = {2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f};

	feed(feats);

	struct med_capture cap = capture();

	zassert_equal(bin(&cap, ROW_MEAN, 2), 1, "values >p99 must clamp to 1.0, not exceed it");
	zassert_equal(bin(&cap, ROW_MAX, MED_BINS - 1), 1);
	zassert_equal(bin(&cap, ROW_FLOOR, 2), 1);
}

/* Each update increments exactly one bin per row. */
ZTEST(obsv_med, test_one_increment_per_row_per_update)
{
	const float a[MED_FEATURES] = {0};
	const float b[MED_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(a);
	feed(b);

	struct med_capture cap = capture();

	for (uint16_t r = 0; r < MED_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 2, "row %u must total the update count", r);
	}
}

/* Empty feature vector (n == 0) is a no-op. */
ZTEST(obsv_med, test_empty_vector_is_noop)
{
	const float feats[MED_FEATURES] = {0};

	zassert_ok(nrf_edgeai_obsv_core_update_features(&ctx, feats, 0));

	struct med_capture cap = capture();

	for (uint16_t r = 0; r < MED_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 0);
	}
}

/* reset() zeroes every row. */
ZTEST(obsv_med, test_reset_clears_histogram)
{
	const float feats[MED_FEATURES] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

	feed(feats);
	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));

	struct med_capture cap = capture();

	for (uint16_t r = 0; r < MED_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 0);
	}
}

/* The metric consumes FEATURES; a probability update must not touch it. */
ZTEST(obsv_med, test_ignores_probs_stream)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	zassert_ok(nrf_edgeai_obsv_core_update_probs(&ctx, probs));

	struct med_capture cap = capture();

	for (uint16_t r = 0; r < MED_ROWS; r++) {
		zassert_equal(row_total(&cap, r), 0,
			"FEATURES metric must ignore the PROBS stream");
	}
}
