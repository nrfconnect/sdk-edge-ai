/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file suite_user_storage.c
 * @brief Tests for nrf_edgeai_obsv_metric_tm_create / _pd_create with
 *        caller-provided storage buffers.
 */

#include <stdint.h>
#include <string.h>

#include <zephyr/ztest.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "common.h"

/* Use a deliberately small class count so the user-storage tests are independent
 * of TEST_NUM_CLASSES and clearly show exact-fit sizing.
 */
#define US_NUM_CLASSES 3U

static nrf_edgeai_obsv_core_t ctx;

static const nrf_edgeai_obsv_model_info_t test_model = {
	.model_id = TEST_MODEL_ID,
	.num_classes = US_NUM_CLASSES,
	.version = TEST_MODEL_VERSION,
};

static void us_before(void *fixture)
{
	ARG_UNUSED(fixture);
	memset(&ctx, 0, sizeof(ctx));
	nrf_edgeai_obsv_core_init(&ctx, &test_model);
}

ZTEST_SUITE(obsv_user_storage, NULL, NULL, us_before, NULL, NULL);

/* --- TM tests ---------------------------------------------------------------- */

ZTEST(obsv_user_storage, test_tm_create_registers_and_snapshots)
{
	static uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(US_NUM_CLASSES) /
			       sizeof(uint32_t)];
	nrf_edgeai_obsv_metric_t tm;

	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, US_NUM_CLASSES);
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &tm, NULL), 0);

	/* Feed two inferences: 0->1, 1->2. */
	const float p0[US_NUM_CLASSES] = {1.0f, 0.0f, 0.0f};
	const float p1[US_NUM_CLASSES] = {0.0f, 1.0f, 0.0f};
	const float p2[US_NUM_CLASSES] = {0.0f, 0.0f, 1.0f};

	nrf_edgeai_obsv_core_update(&ctx, p0);
	nrf_edgeai_obsv_core_update(&ctx, p1);
	nrf_edgeai_obsv_core_update(&ctx, p2);

	struct test_snapshots snaps = {0};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps), 0);
	zassert_true(snaps.transition_matrix.present);
	zassert_equal(snaps.transition_matrix.num_rows, US_NUM_CLASSES);
	zassert_equal(snaps.transition_matrix.num_cols, US_NUM_CLASSES);

	/* Transition 0->1 and 1->2 each happened once. */
	zassert_equal(snaps.transition_matrix.counts[0 * US_NUM_CLASSES + 1], 1U);
	zassert_equal(snaps.transition_matrix.counts[1 * US_NUM_CLASSES + 2], 1U);
}

ZTEST(obsv_user_storage, test_tm_create_clear_resets_counters)
{
	static uint32_t tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(US_NUM_CLASSES) /
			       sizeof(uint32_t)];
	nrf_edgeai_obsv_metric_t tm;

	nrf_edgeai_obsv_metric_tm_create(&tm, tm_buf, US_NUM_CLASSES);
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &tm, NULL), 0);

	const float p0[US_NUM_CLASSES] = {1.0f, 0.0f, 0.0f};
	const float p1[US_NUM_CLASSES] = {0.0f, 1.0f, 0.0f};

	nrf_edgeai_obsv_core_update(&ctx, p0);
	nrf_edgeai_obsv_core_update(&ctx, p1);

	zassert_equal(nrf_edgeai_obsv_core_reset(&ctx), 0);

	struct test_snapshots snaps = {0};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps), 0);
	zassert_true(snaps.transition_matrix.present);

	for (uint32_t i = 0; i < US_NUM_CLASSES * US_NUM_CLASSES; i++) {
		zassert_equal(snaps.transition_matrix.counts[i], 0U,
			      "TM counter [%u] not zero after reset", i);
	}
}

/* --- PD tests ---------------------------------------------------------------- */

ZTEST(obsv_user_storage, test_pd_create_registers_and_snapshots)
{
	static uint32_t pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(US_NUM_CLASSES) /
			       sizeof(uint32_t)];
	nrf_edgeai_obsv_metric_t pd;

	nrf_edgeai_obsv_metric_pd_create(&pd, pd_buf, US_NUM_CLASSES);
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &pd, NULL), 0);

	/* All classes at 1.0 — should land in the last bin for each class. */
	const float p[US_NUM_CLASSES] = {1.0f, 1.0f, 1.0f};

	nrf_edgeai_obsv_core_update(&ctx, p);

	struct test_snapshots snaps = {0};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps), 0);
	zassert_true(snaps.probs_distribution.present);
	zassert_equal(snaps.probs_distribution.num_rows, US_NUM_CLASSES);
	zassert_equal(snaps.probs_distribution.num_cols, TEST_NUM_BINS);

	/* Each class had exactly one inference at 1.0 — last bin must be 1. */
	const uint32_t last_bin = TEST_NUM_BINS - 1;

	for (uint32_t cls = 0; cls < US_NUM_CLASSES; cls++) {
		zassert_equal(snaps.probs_distribution.counts[cls * TEST_NUM_BINS + last_bin], 1U,
			      "class %u last bin not 1", cls);
	}
}

ZTEST(obsv_user_storage, test_pd_create_custom_edges)
{
	static uint32_t pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(US_NUM_CLASSES) /
			       sizeof(uint32_t)];
	nrf_edgeai_obsv_metric_t pd;
	nrf_obsv_probs_dist_cfg_t cfg = {0};

	/* Build uniform inner edges explicitly and pass as cfg (0.0 and 1.0 are implicit). */
	const uint32_t n_bins = TEST_NUM_BINS;
	const float step = 1.0f / n_bins;

	for (uint32_t i = 0; i < n_bins - 1; i++) {
		cfg.bin_edges[i] = (i + 1) * step;
	}

	nrf_edgeai_obsv_metric_pd_create(&pd, pd_buf, US_NUM_CLASSES);
	zassert_equal(nrf_edgeai_obsv_core_register(&ctx, &pd, &cfg), 0);

	const float p[US_NUM_CLASSES] = {0.0f, 0.5f, 1.0f};

	nrf_edgeai_obsv_core_update(&ctx, p);

	struct test_snapshots snaps = {0};

	zassert_equal(nrf_edgeai_obsv_core_for_each_metric(&ctx, test_capture_cb, &snaps), 0);
	zassert_true(snaps.probs_distribution.present);

	/* Class 0 at 0.0 → bin 0; class 2 at 1.0 → last bin. */
	zassert_equal(snaps.probs_distribution.counts[0 * TEST_NUM_BINS + 0], 1U);
	zassert_equal(snaps.probs_distribution.counts[2 * TEST_NUM_BINS + (TEST_NUM_BINS - 1)], 1U);
}
