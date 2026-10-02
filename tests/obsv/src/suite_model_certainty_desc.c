/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define MCD_BINS CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM

/* Rows of the 3 x bin_num matrix. */
#define ROW_ENTROPY   0U
#define ROW_MARGIN    1U
#define ROW_STABILITY 2U

/* Columns of the stability row. */
#define STAB_SWITCHES	      0U
#define STAB_COMPARISONS      1U
#define STAB_MAJORITY_FRAMES  2U
#define STAB_CONFIDENT_SWITCH 3U

struct mcd_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t m[NRF_EDGEAI_OBSV_MCD_NUM_ROWS * 16];
};

static bool mcd_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct mcd_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_MODEL_CERTAINTY_DESC) {
		return true;
	}

	cap->present = true;
	cap->metric_id = snap->metric_id;
	cap->version = snap->version;
	cap->num_rows = snap->num_rows;
	cap->num_cols = snap->num_cols;

	size_t n = (size_t)snap->num_rows * snap->num_cols;

	for (size_t i = 0; i < n && i < ARRAY_SIZE(cap->m); i++) {
		cap->m[i] = snap->counts[i];
	}

	return true;
}

static nrf_edgeai_obsv_core_t ctx;
static uint32_t mcd_buf[NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(TEST_NUM_CLASSES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t mcd_metric;

static void mcd_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_mcd_create(&mcd_metric, mcd_buf, TEST_NUM_CLASSES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &mcd_metric, NULL));
}

static struct mcd_capture capture(void)
{
	struct mcd_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, mcd_capture_cb, &cap));
	zassert_true(cap.present, "model certainty descriptor snapshot not visited");

	return cap;
}

/* Cell [row][col] of the captured matrix. */
static uint32_t cell(const struct mcd_capture *cap, uint16_t row, uint16_t col)
{
	return cap->m[(size_t)row * cap->num_cols + col];
}

/* Sum of one histogram row. */
static uint32_t row_total(const struct mcd_capture *cap, uint16_t row)
{
	uint32_t sum = 0;

	for (uint16_t i = 0; i < cap->num_cols; i++) {
		sum += cell(cap, row, i);
	}
	return sum;
}

/* Feed one probability vector of TEST_NUM_CLASSES entries. */
static void feed(const float *probs)
{
	zassert_ok(nrf_edgeai_obsv_core_update_probs(&ctx, probs));
}

/*
 * Build a probability vector whose argmax is @p cls and whose winning
 * probability is @p top, with the remaining mass spread over the other classes.
 */
static void feed_cls(uint16_t cls, float top)
{
	float probs[TEST_NUM_CLASSES];
	float rest = (1.0f - top) / (float)(TEST_NUM_CLASSES - 1);

	for (uint16_t i = 0; i < TEST_NUM_CLASSES; i++) {
		probs[i] = rest;
	}
	probs[cls] = top;

	feed(probs);
}

/* Feed a sequence of argmax classes, each as a decisive (0.9) majority winner. */
static void feed_seq(const uint16_t *classes, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		feed_cls(classes[i], 0.9f);
	}
}

ZTEST_SUITE(obsv_mcd, NULL, NULL, mcd_setup, NULL, NULL);

/* Snapshot identity and shape: id 1, version 1, 3 x bin_num. */
ZTEST(obsv_mcd, test_snapshot_shape)
{
	struct mcd_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_MODEL_CERTAINTY_DESC);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.num_rows, NRF_EDGEAI_OBSV_MCD_NUM_ROWS);
	zassert_equal(cap.num_cols, MCD_BINS);
}

/* --- row 0: normalized-entropy histogram --- */

/* One-hot vector -> H = 0 -> normalized 0 -> lowest bin. */
ZTEST(obsv_mcd, test_entropy_one_hot_is_min)
{
	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};

	feed(probs);

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_ENTROPY, 0), 1, "zero entropy must land in bin 0");
	zassert_equal(row_total(&cap, ROW_ENTROPY), 1);
}

/* Uniform vector -> H = ln(N) -> normalized 1 -> top bin. */
ZTEST(obsv_mcd, test_entropy_uniform_is_max)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	feed(probs);

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_ENTROPY, MCD_BINS - 1), 1, "max entropy must land in top bin");
	zassert_equal(row_total(&cap, ROW_ENTROPY), 1);
}

/*
 * H([0.7,0.1,0.1,0.1]) ~= 0.940 nats; normalized ~= 0.940 / ln(4) ~= 0.679,
 * which falls in [0.5, 0.75) -> bin 2 of 4 uniform bins.
 */
ZTEST(obsv_mcd, test_entropy_mid_lands_in_middle_bin)
{
	const float probs[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};

	feed(probs);

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_ENTROPY, 2), 1, "normalized entropy ~0.68 must land in bin 2");
	zassert_equal(row_total(&cap, ROW_ENTROPY), 1);
}

/* --- row 1: top-2 margin histogram --- */

/* One-hot vector -> margin = 1 -> top bin (most decisive). */
ZTEST(obsv_mcd, test_margin_one_hot_is_max)
{
	const float probs[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};

	feed(probs);

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_MARGIN, MCD_BINS - 1), 1, "max margin must land in top bin");
	zassert_equal(row_total(&cap, ROW_MARGIN), 1);
}

/* Uniform vector -> top1 == top2 -> margin = 0 -> bin 0. */
ZTEST(obsv_mcd, test_margin_uniform_is_min)
{
	const float probs[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};

	feed(probs);

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_MARGIN, 0), 1, "zero margin must land in bin 0");
	zassert_equal(row_total(&cap, ROW_MARGIN), 1);
}

/* margin([0.8, 0.2, 0.0, 0.0]) = 0.6 -> [0.5, 0.75) -> bin 2 of 4. */
ZTEST(obsv_mcd, test_margin_mid_lands_in_middle_bin)
{
	const float probs[TEST_NUM_CLASSES] = {0.8f, 0.2f, 0.0f, 0.0f};

	feed(probs);

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_MARGIN, 2), 1, "margin 0.6 must land in bin 2");
	zassert_equal(row_total(&cap, ROW_MARGIN), 1);
}

/* --- row 2: stability counters --- */

/* No comparison is possible before a second inference arrives. */
ZTEST(obsv_mcd, test_stability_initial_state)
{
	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_STABILITY, STAB_SWITCHES), 0);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 0);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_MAJORITY_FRAMES), 0);

	feed_cls(2, 0.9f);
	cap = capture();
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_SWITCHES), 0,
		      "first inference must not count a comparison");
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 0);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_MAJORITY_FRAMES), 1,
		      "a p_top1 > 0.5 inference is a majority frame");
}

/*
 * Sequence 0,0,1,1,2,0 (all decisive) -> 5 pairs, 3 switch class.
 * Every frame is a majority winner and every switch is into a majority winner.
 */
ZTEST(obsv_mcd, test_stability_counts_match_formula)
{
	const uint16_t seq[] = {0, 0, 1, 1, 2, 0};

	feed_seq(seq, ARRAY_SIZE(seq));

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 5,
		      "expected N-1 = 5 comparisons");
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_SWITCHES), 3, "expected 3 class switches");
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_MAJORITY_FRAMES), 6);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_CONFIDENT_SWITCH), 3);
}

/* A run of identical classes never switches. */
ZTEST(obsv_mcd, test_stability_stable_stream)
{
	const uint16_t seq[] = {1, 1, 1, 1};

	feed_seq(seq, ARRAY_SIZE(seq));

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 3);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_SWITCHES), 0);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_CONFIDENT_SWITCH), 0);
}

/* Alternating classes switch on every pair. */
ZTEST(obsv_mcd, test_stability_alternating_stream)
{
	const uint16_t seq[] = {0, 1, 0, 1, 0};

	feed_seq(seq, ARRAY_SIZE(seq));

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 4);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_SWITCHES), 4);
}

/* majority_frames counts only inferences whose winner exceeds 0.5. */
ZTEST(obsv_mcd, test_stability_majority_frames)
{
	feed_cls(0, 0.4f); /* top1 = 0.4, not a majority */
	feed_cls(0, 0.9f); /* top1 = 0.9, a majority */

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_STABILITY, STAB_MAJORITY_FRAMES), 1);
}

/*
 * confident_switches counts only switches into a majority (>0.5) winner.
 * 0@0.9 -> 1@0.4 (switch, new winner not majority) -> 0@0.9 (switch into majority):
 * 2 switches, 1 confident.
 */
ZTEST(obsv_mcd, test_stability_confident_switches)
{
	feed_cls(0, 0.9f);
	feed_cls(1, 0.4f);
	feed_cls(0, 0.9f);

	struct mcd_capture cap = capture();

	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 2);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_SWITCHES), 2);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_CONFIDENT_SWITCH), 1);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_MAJORITY_FRAMES), 2);
}

/* Every update increments exactly one bin in each histogram row. */
ZTEST(obsv_mcd, test_histograms_total_equals_update_count)
{
	const float a[TEST_NUM_CLASSES] = {1.0f, 0.0f, 0.0f, 0.0f};
	const float b[TEST_NUM_CLASSES] = {0.25f, 0.25f, 0.25f, 0.25f};
	const float c[TEST_NUM_CLASSES] = {0.7f, 0.1f, 0.1f, 0.1f};

	feed(a);
	feed(b);
	feed(c);

	struct mcd_capture cap = capture();

	zassert_equal(row_total(&cap, ROW_ENTROPY), 3);
	zassert_equal(row_total(&cap, ROW_MARGIN), 3);
}

/* reset() zeroes every row, including the stability state. */
ZTEST(obsv_mcd, test_reset_clears_everything)
{
	const uint16_t seq[] = {0, 1, 2};

	feed_seq(seq, ARRAY_SIZE(seq));
	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));

	struct mcd_capture cap = capture();

	zassert_equal(row_total(&cap, ROW_ENTROPY), 0);
	zassert_equal(row_total(&cap, ROW_MARGIN), 0);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_SWITCHES), 0);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 0);
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_MAJORITY_FRAMES), 0);

	/* After reset the next single inference still records no comparison. */
	feed_cls(3, 0.9f);
	cap = capture();
	zassert_equal(cell(&cap, ROW_STABILITY, STAB_COMPARISONS), 0,
		      "reset must forget the previous class");
}
