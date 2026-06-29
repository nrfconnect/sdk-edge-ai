/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "common.h"

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

/* Counter column order exported by the metric snapshot. */
#define IDX_SWITCHES	0U
#define IDX_COMPARISONS 1U

struct psr_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t switches;
	uint32_t comparisons;
};

static bool psr_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct psr_capture *cap = user;

	if (snap->metric_id != NRF_EDGEAI_OBSV_METRIC_ID_PREDICTION_SWITCHING_RATE) {
		return true;
	}

	cap->present = true;
	cap->metric_id = snap->metric_id;
	cap->version = snap->version;
	cap->num_rows = snap->num_rows;
	cap->num_cols = snap->num_cols;
	cap->switches = snap->counts[IDX_SWITCHES];
	cap->comparisons = snap->counts[IDX_COMPARISONS];

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
static uint32_t psr_buf[NRF_EDGEAI_OBSV_PSR_STORAGE_BYTES(TEST_NUM_CLASSES) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t psr_metric;

static void psr_setup(void *fixture)
{
	ARG_UNUSED(fixture);

	const nrf_edgeai_obsv_model_info_t model = {
		.model_id = TEST_MODEL_ID,
		.num_classes = TEST_NUM_CLASSES,
		.version = TEST_MODEL_VERSION,
	};

	zassert_ok(nrf_edgeai_obsv_core_init(&ctx, &model));

	nrf_edgeai_obsv_metric_psr_create(&psr_metric, psr_buf, TEST_NUM_CLASSES);
	zassert_ok(nrf_edgeai_obsv_core_register(&ctx, &psr_metric, NULL));
}

static void feed(const uint16_t *classes, size_t n)
{
	float probs[TEST_NUM_CLASSES];

	for (size_t i = 0; i < n; i++) {
		make_probs(probs, classes[i]);
		zassert_ok(nrf_edgeai_obsv_core_update_probs(&ctx, probs));
	}
}

static struct psr_capture capture(void)
{
	struct psr_capture cap = {0};

	zassert_ok(nrf_edgeai_obsv_core_for_each_metric(&ctx, psr_capture_cb, &cap));
	zassert_true(cap.present, "PSR metric snapshot not visited");

	return cap;
}

ZTEST_SUITE(obsv_psr, NULL, NULL, psr_setup, NULL, NULL);

/* Snapshot shape and identity: fixed 1 x 2 row, id 4, version 1. */
ZTEST(obsv_psr, test_snapshot_shape)
{
	struct psr_capture cap = capture();

	zassert_equal(cap.metric_id, NRF_EDGEAI_OBSV_METRIC_ID_PREDICTION_SWITCHING_RATE);
	zassert_equal(cap.version, 1);
	zassert_equal(cap.num_rows, 1);
	zassert_equal(cap.num_cols, 2);
}

/* No comparison is possible before a second inference arrives. */
ZTEST(obsv_psr, test_initial_state_is_zero)
{
	struct psr_capture cap = capture();

	zassert_equal(cap.switches, 0);
	zassert_equal(cap.comparisons, 0);

	const uint16_t single[] = {2};

	feed(single, ARRAY_SIZE(single));
	cap = capture();
	zassert_equal(cap.switches, 0, "first inference must not count a comparison");
	zassert_equal(cap.comparisons, 0);
}

/*
 * Sequence 0,0,1,1,2,0 -> 5 consecutive pairs, 3 of which switch class.
 * SwitchRate = switches / comparisons = 3 / 5.
 */
ZTEST(obsv_psr, test_counts_match_formula)
{
	const uint16_t seq[] = {0, 0, 1, 1, 2, 0};

	feed(seq, ARRAY_SIZE(seq));

	struct psr_capture cap = capture();

	zassert_equal(cap.comparisons, 5, "expected N-1 = 5 comparisons");
	zassert_equal(cap.switches, 3, "expected 3 class switches");
}

/* A run of identical classes never switches. */
ZTEST(obsv_psr, test_stable_stream_has_no_switches)
{
	const uint16_t seq[] = {1, 1, 1, 1};

	feed(seq, ARRAY_SIZE(seq));

	struct psr_capture cap = capture();

	zassert_equal(cap.comparisons, 3);
	zassert_equal(cap.switches, 0);
}

/* Alternating classes switch on every pair. */
ZTEST(obsv_psr, test_alternating_stream_switches_every_pair)
{
	const uint16_t seq[] = {0, 1, 0, 1, 0};

	feed(seq, ARRAY_SIZE(seq));

	struct psr_capture cap = capture();

	zassert_equal(cap.comparisons, 4);
	zassert_equal(cap.switches, 4);
}

/* reset() clears counters and the remembered previous class. */
ZTEST(obsv_psr, test_reset_clears_counters)
{
	const uint16_t seq[] = {0, 1, 2};

	feed(seq, ARRAY_SIZE(seq));
	zassert_ok(nrf_edgeai_obsv_core_reset(&ctx));

	struct psr_capture cap = capture();

	zassert_equal(cap.switches, 0);
	zassert_equal(cap.comparisons, 0);

	/* After reset the next single inference still records no comparison. */
	const uint16_t single[] = {3};

	feed(single, ARRAY_SIZE(single));
	cap = capture();
	zassert_equal(cap.comparisons, 0, "reset must forget the previous class");
}
