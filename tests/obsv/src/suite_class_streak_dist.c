/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define CSD_BINS CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_BIN_NUM
#define CSD_TOP	 CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOP
#define CSD_TOL	 CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOLERANCE

/*
 * The bin/length arithmetic below is pinned to a specific configuration so the
 * expected bins are exact. Streak length L is normalised to (L-1)/(TOP-1) and
 * binned uniformly over [0, 1]; with TOP=5 and 4 bins that is:
 *   L=1 -> bin 0, L=2 -> bin 1, L=3 -> bin 2, L=4 -> bin 3, L>=5 -> top bin (3).
 * TOLERANCE=1 means a single consecutive mismatch is bridged (not counted into
 * the length) while two consecutive mismatches end the streak.
 */
BUILD_ASSERT(CSD_BINS == 4, "csd suite assumes 4 bins");
BUILD_ASSERT(CSD_TOP == 5, "csd suite assumes TOP=5 (L->bin: 1->0,2->1,3->2,4->3,>=5->top)");
BUILD_ASSERT(CSD_TOL == 1, "csd suite assumes TOLERANCE=1");

struct csd_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t counts[TEST_NUM_CLASSES * 16];
};

static bool csd_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct csd_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_CLASS_STREAK_DIST) {
		return true;
	}

	cap->present = true;
	cap->metric_id = snap->metric_id;
	cap->version = snap->version;
	cap->num_rows = snap->num_rows;
	cap->num_cols = snap->num_cols;

	const size_t cells = (size_t)snap->num_rows * snap->num_cols;

	for (size_t i = 0; i < cells && i < ARRAY_SIZE(cap->counts); i++) {
		cap->counts[i] = snap->counts[i];
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
static uint32_t csd_buf[NRF_EDGEAI_OBSV_CSD_STORAGE_BYTES(TEST_NUM_CLASSES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t csd_metric;

static void csd_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_csd_create(&csd_metric, csd_buf, TEST_NUM_CLASSES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &csd_metric, NULL));
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

static struct csd_capture capture(void)
{
	struct csd_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, csd_capture_cb, &cap));
	zassert_true(cap.present, "CSD metric snapshot not visited");

	return cap;
}

static uint32_t cell(const struct csd_capture *cap, uint16_t row, uint16_t col)
{
	return cap->counts[(size_t)row * cap->num_cols + col];
}

static uint32_t total(const struct csd_capture *cap)
{
	uint32_t sum = 0;

	for (size_t i = 0; i < (size_t)cap->num_rows * cap->num_cols; i++) {
		sum += cap->counts[i];
	}
	return sum;
}

ZTEST_SUITE(obsv_csd, NULL, NULL, csd_setup, NULL, NULL);

/* Snapshot shape and identity: num_classes x bin_num, id 9, version 1. */
ZTEST(obsv_csd, test_snapshot_shape)
{
	struct csd_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_CLASS_STREAK_DIST);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.num_rows, TEST_NUM_CLASSES);
	zassert_equal(cap.num_cols, CSD_BINS);
	zassert_equal(total(&cap), 0, "no streak has completed yet");
}

/* create() stores the configured dimensions/config in the storage header. */
ZTEST(obsv_csd, test_header_config)
{
	const _nrf_obsv_csd_hdr_t *h = (const _nrf_obsv_csd_hdr_t *)csd_buf;

	zassert_equal(h->num_classes, TEST_NUM_CLASSES);
	zassert_equal(h->bin_num, CSD_BINS);
	zassert_equal(h->top, CSD_TOP);
	zassert_equal(h->tolerance, CSD_TOL);
}

/* A streak still in progress is not recorded until it ends. */
ZTEST(obsv_csd, test_active_streak_not_recorded)
{
	const uint16_t seq[] = {0, 0, 0, 0, 0};

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(total(&cap), 0, "an unfinished streak must not be binned");
}

/*
 * Streak length maps to the expected bin (TOP=5, 4 bins). Each streak is closed
 * by two consecutive mismatches (TOLERANCE=1: one bridged, the second breaks).
 */
ZTEST(obsv_csd, test_length_one_lands_in_bin0)
{
	const uint16_t seq[] = {0, 1, 1}; /* class 0 held for 1 frame, then broken */

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 0), 1, "length-1 streak must land in bin 0");
	zassert_equal(total(&cap), 1);
}

ZTEST(obsv_csd, test_length_two_lands_in_bin1)
{
	const uint16_t seq[] = {0, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 1), 1, "length-2 streak must land in bin 1");
	zassert_equal(total(&cap), 1);
}

ZTEST(obsv_csd, test_length_three_lands_in_bin2)
{
	const uint16_t seq[] = {0, 0, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 2), 1, "length-3 streak must land in bin 2");
	zassert_equal(total(&cap), 1);
}

ZTEST(obsv_csd, test_length_four_lands_in_top_bin)
{
	const uint16_t seq[] = {0, 0, 0, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(cell(&cap, 0, CSD_BINS - 1), 1, "length-4 streak must land in the top bin");
	zassert_equal(total(&cap), 1);
}

/* A run longer than TOP saturates (caps at TOP) and still lands in the top bin. */
ZTEST(obsv_csd, test_long_streak_caps_in_top_bin)
{
	const uint16_t seq[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2}; /* 10x class 0 */

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(cell(&cap, 0, CSD_BINS - 1), 1, "capped streak must land in the top bin");
	zassert_equal(total(&cap), 1);
}

/*
 * A single interposed mismatch is bridged (TOLERANCE=1) and is NOT counted into
 * the length: {0,0,1,0,2,2} is a class-0 streak of length 3 (three 0s; the lone
 * 1 is bridged), so it lands in bin 2 — not bin 3, which is where length 4 (the
 * value it would have had if the bridged frame were counted) would go.
 */
ZTEST(obsv_csd, test_single_mismatch_is_bridged_not_counted)
{
	const uint16_t seq[] = {0, 0, 1, 0, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 2), 1, "bridged flicker must not add to the length");
	zassert_equal(total(&cap), 1);
}

/*
 * The tolerance boundary: one mismatch only bridges (nothing recorded yet); the
 * second consecutive mismatch ends the streak and records it.
 */
ZTEST(obsv_csd, test_two_consecutive_mismatches_break)
{
	const uint16_t run[] = {0, 0, 0};

	feed(run, ARRAY_SIZE(run));

	struct csd_capture cap = capture();

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
ZTEST(obsv_csd, test_break_seeds_new_streak_and_rows_are_independent)
{
	const uint16_t seq[] = {0, 0, 1, 1, 1, 1, 2, 2};

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

	zassert_equal(cell(&cap, 0, 1), 1, "class 0: length-2 streak in bin 1");
	zassert_equal(cell(&cap, 1, 2), 1, "class 1 seeded by breaking frame: len-3 in bin 2");
	zassert_equal(total(&cap), 2, "exactly two streaks completed");
}

/* reset() zeroes counters and clears the in-progress streak state. */
ZTEST(obsv_csd, test_reset_clears_counters_and_state)
{
	const uint16_t seq[] = {0, 0, 1, 1};

	feed(seq, ARRAY_SIZE(seq));

	struct csd_capture cap = capture();

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
