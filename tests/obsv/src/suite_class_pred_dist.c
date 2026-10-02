/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define CPD_BINS CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM
#define CPD_TOP	 CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOP_BIN
#define CPD_TOL	 CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOL

/*
 * The class predictions distribution is a 2*num_classes x bin_num matrix: rows
 * [0, N) the probability distribution, rows [N, 2N) the streak distribution. This
 * suite exercises the streak half (the probability half is covered by
 * suite_payload.c); the capture callback below copies the streak sub-block so
 * cell(row, col) addresses a class's streak-length histogram.
 *
 * The bin/length arithmetic is pinned to a specific configuration so the expected
 * bins are exact. Streak length L is normalised to (L-1)/(TOP-1) and binned
 * uniformly over [0, 1]; with TOP=5 and 4 bins that is:
 *   L=1 -> bin 0, L=2 -> bin 1, L=3 -> bin 2, L=4 -> bin 3, L>=5 -> top bin (3).
 * TOLERANCE=1 means a single consecutive mismatch is bridged (not counted into
 * the length) while two consecutive mismatches end the streak.
 */
BUILD_ASSERT(CPD_BINS == 4, "cpd suite assumes 4 bins");
BUILD_ASSERT(CPD_TOP == 5, "cpd suite assumes TOP=5 (L->bin: 1->0,2->1,3->2,4->3,>=5->top)");
BUILD_ASSERT(CPD_TOL == 1, "cpd suite assumes TOLERANCE=1");

struct cpd_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t full_rows; /* rows of the whole matrix (2 * num_classes) */
	uint16_t num_rows;  /* rows of the streak sub-block (num_classes) */
	uint16_t num_cols;
	uint32_t counts[TEST_NUM_CLASSES * 16]; /* streak sub-block only */
};

static bool cpd_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct cpd_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_CLASS_PRED_DIST) {
		return true;
	}

	uint16_t n = snap->num_rows / 2U; /* per-class rows in each sub-block */

	cap->present = true;
	cap->metric_id = snap->metric_id;
	cap->version = snap->version;
	cap->full_rows = snap->num_rows;
	cap->num_rows = n;
	cap->num_cols = snap->num_cols;

	/* Copy the streak sub-block (rows [n, 2n)). */
	const size_t off = (size_t)n * snap->num_cols;
	const size_t cells = (size_t)n * snap->num_cols;

	for (size_t i = 0; i < cells && i < ARRAY_SIZE(cap->counts); i++) {
		cap->counts[i] = snap->counts[off + i];
	}

	return true;
}

/* Build a probability vector of TEST_NUM_CLASSES entries whose argmax is @p cls. */
static void make_probs(float *probs, uint16_t cls)
{
	for (uint16_t i = 0; i < TEST_NUM_CLASSES; i++) {
		probs[i] = 0.1f;
	}
	probs[cls] = 0.9f;
}

static nrf_edgeai_obsv_core_t ctx;
static uint32_t cpd_buf[NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(TEST_NUM_CLASSES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t cpd_metric;

static void cpd_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_cpd_create(&cpd_metric, cpd_buf, TEST_NUM_CLASSES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &cpd_metric, NULL));
}

/* Feed a sequence of argmax classes, one inference per entry. */
static void feed(const uint16_t *classes, size_t n)
{
	float probs[TEST_NUM_CLASSES];

	for (size_t i = 0; i < n; i++) {
		make_probs(probs, classes[i]);
		zassert_ok(nrf_edgeai_obsv_core_update_probs(&ctx, probs));
	}
}

static struct cpd_capture capture(void)
{
	struct cpd_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, cpd_capture_cb, &cap));
	zassert_true(cap.present, "class predictions distribution snapshot not visited");

	return cap;
}

/* A cell of the streak sub-block: row = class index, col = streak-length bin. */
static uint32_t cell(const struct cpd_capture *cap, uint16_t row, uint16_t col)
{
	return cap->counts[(size_t)row * cap->num_cols + col];
}

static uint32_t total(const struct cpd_capture *cap)
{
	uint32_t sum = 0;

	for (size_t i = 0; i < (size_t)cap->num_rows * cap->num_cols; i++) {
		sum += cap->counts[i];
	}
	return sum;
}

ZTEST_SUITE(obsv_cpd, NULL, NULL, cpd_setup, NULL, NULL);

/* Snapshot shape and identity: 2*num_classes x bin_num, id 2, version 1. */
ZTEST(obsv_cpd, test_snapshot_shape)
{
	struct cpd_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_CLASS_PRED_DIST);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.full_rows, 2 * TEST_NUM_CLASSES);
	zassert_equal(cap.num_cols, CPD_BINS);
	zassert_equal(total(&cap), 0, "no streak has completed yet");
}

/* create() stores the configured dimensions/config in the storage header. */
ZTEST(obsv_cpd, test_header_config)
{
	const _nrf_obsv_cpd_hdr_t *h = (const _nrf_obsv_cpd_hdr_t *)cpd_buf;

	zassert_equal(h->num_classes, TEST_NUM_CLASSES);
	zassert_equal(h->bin_num, CPD_BINS);
	zassert_equal(h->cfg[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOP], CPD_TOP);
	zassert_equal(h->cfg[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOL], CPD_TOL);
}

/* The snapshot reports the streak top and tolerance the counters were gathered with. */
ZTEST(obsv_cpd, test_snapshot_reports_config)
{
	struct test_snapshots snaps = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps));
	zassert_true(snaps.probs_distribution.present);
	zassert_equal(snaps.probs_distribution.config_rows, 1);
	zassert_equal(snaps.probs_distribution.config_cols, NRF_EDGEAI_OBSV_CPD_CFG_COUNT);
	zassert_equal(snaps.probs_distribution.config[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOP], CPD_TOP);
	zassert_equal(snaps.probs_distribution.config[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOL], CPD_TOL);
}

/* The probability sub-block accumulates one sample per class per inference. */
ZTEST(obsv_cpd, test_probability_rows_sum_to_n)
{
	const uint16_t seq[] = {0, 1, 2};

	feed(seq, ARRAY_SIZE(seq));

	/* Read the probability sub-block directly (rows [0, N)). */
	struct test_snapshots snaps = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps));
	zassert_true(snaps.probs_distribution.present);
	zassert_equal(snaps.probs_distribution.num_rows, TEST_NUM_CLASSES);

	const struct test_metric_capture *pd = &snaps.probs_distribution;

	for (uint16_t c = 0; c < pd->num_rows; c++) {
		uint32_t row_sum = 0;

		for (uint16_t b = 0; b < pd->num_cols; b++) {
			row_sum += pd->counts[(size_t)c * pd->num_cols + b];
		}
		zassert_equal(row_sum, ARRAY_SIZE(seq),
			      "every probability row must total the inference count");
	}
}

/* A streak still in progress is not recorded until it ends. */
ZTEST(obsv_cpd, test_active_streak_not_recorded)
{
	const uint16_t seq[] = {0, 0, 0, 0, 0};

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(total(&cap), 0, "an unfinished streak must not be binned");
}

/*
 * Streak length maps to the expected bin (TOP=5, 4 bins). Each streak is closed
 * by two consecutive mismatches (TOLERANCE=1: one bridged, the second breaks).
 */
ZTEST(obsv_cpd, test_length_one_lands_in_bin0)
{
	const uint16_t seq[] = {0, 1, 1}; /* class 0 held for 1 frame, then broken */

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 0), 1, "length-1 streak must land in bin 0");
	zassert_equal(total(&cap), 1);
}

ZTEST(obsv_cpd, test_length_two_lands_in_bin1)
{
	const uint16_t seq[] = {0, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 1), 1, "length-2 streak must land in bin 1");
	zassert_equal(total(&cap), 1);
}

ZTEST(obsv_cpd, test_length_three_lands_in_bin2)
{
	const uint16_t seq[] = {0, 0, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 2), 1, "length-3 streak must land in bin 2");
	zassert_equal(total(&cap), 1);
}

ZTEST(obsv_cpd, test_length_four_lands_in_top_bin)
{
	const uint16_t seq[] = {0, 0, 0, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(cell(&cap, 0, CPD_BINS - 1), 1, "length-4 streak must land in the top bin");
	zassert_equal(total(&cap), 1);
}

/* A run longer than TOP saturates (caps at TOP) and still lands in the top bin. */
ZTEST(obsv_cpd, test_long_streak_caps_in_top_bin)
{
	const uint16_t seq[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2}; /* 10x class 0 */

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(cell(&cap, 0, CPD_BINS - 1), 1, "capped streak must land in the top bin");
	zassert_equal(total(&cap), 1);
}

/*
 * A single interposed mismatch is bridged (TOLERANCE=1) and is NOT counted into
 * the length: {0,0,1,0,2,2} is a class-0 streak of length 3 (three 0s; the lone
 * 1 is bridged), so it lands in bin 2 — not bin 3, which is where length 4 (the
 * value it would have had if the bridged frame were counted) would go.
 */
ZTEST(obsv_cpd, test_single_mismatch_is_bridged_not_counted)
{
	const uint16_t seq[] = {0, 0, 1, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 2), 1, "bridged flicker must not add to the length");
	zassert_equal(total(&cap), 1);
}

/*
 * The tolerance boundary: one mismatch only bridges (nothing recorded yet); the
 * second consecutive mismatch ends the streak and records it.
 */
ZTEST(obsv_cpd, test_two_consecutive_mismatches_break)
{
	const uint16_t run[] = {0, 0, 0};

	feed(run, ARRAY_SIZE(run));

	struct cpd_capture cap = capture();

	zassert_equal(total(&cap), 0, "active streak: nothing recorded");

	const uint16_t one_miss[] = {1};

	feed(one_miss, ARRAY_SIZE(one_miss));
	cap = capture();
	zassert_equal(total(&cap), 0, "a single mismatch is only bridged, not a break");

	const uint16_t second_miss[] = {1};

	feed(second_miss, ARRAY_SIZE(second_miss));
	cap = capture();

	zassert_equal(cell(&cap, 0, 2), 1, "2nd consecutive mismatch closes the len-3 streak");
	zassert_equal(total(&cap), 1);
}

/*
 * The frame that breaks a streak seeds a new one for its own class, and rows are
 * independent. {0,0,1,1,1,1,2,2}:
 *   - 0,0 then bridged 1 then breaking 1 -> class 0 streak length 2 (bin 1);
 *   - the breaking 1 seeds class 1, extended by two more 1s -> length 3 (bin 2);
 *   - bridged 2 then breaking 2 closes it.
 */
ZTEST(obsv_cpd, test_break_seeds_new_streak_and_rows_are_independent)
{
	const uint16_t seq[] = {0, 0, 1, 1, 1, 1, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 1), 1, "class 0: length-2 streak in bin 1");
	zassert_equal(cell(&cap, 1, 2), 1, "class 1 seeded by breaking frame: len-3 in bin 2");
	zassert_equal(total(&cap), 2, "exactly two streaks completed");
}

/* reset() zeroes counters and clears the in-progress streak state. */
ZTEST(obsv_cpd, test_reset_clears_counters_and_state)
{
	const uint16_t seq[] = {0, 0, 1, 1};

	feed(seq, ARRAY_SIZE(seq));

	struct cpd_capture cap = capture();

	zassert_equal(total(&cap), 1, "sanity: one streak recorded before reset");

	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));
	cap = capture();
	zassert_equal(total(&cap), 0, "reset must zero the histogram");

	/* If reset did not clear the active class, the leading 2 would be treated as a
	 * mismatch against the pre-reset class instead of seeding a fresh streak.
	 */
	const uint16_t after[] = {2, 2, 2, 3, 3};

	feed(after, ARRAY_SIZE(after));
	cap = capture();

	zassert_equal(cell(&cap, 2, 2), 1, "post-reset class-2 length-3 streak in bin 2");
	zassert_equal(total(&cap), 1, "reset must forget the previous streak state");
}
