/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef TEST_OBSV_COMMON_H
#define TEST_OBSV_COMMON_H

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/ztest.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

#define TEST_MODEL_ID	   7
#define TEST_MODEL_VERSION 1
#define TEST_NUM_CLASSES   CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES
#define TEST_NUM_BINS	   CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM

/* counts[] in test_metric_capture is sized for the largest possible metric:
 * TM = MAX_CLASSES x MAX_CLASSES, PD = MAX_CLASSES x BIN_NUM.
 * If BIN_NUM ever exceeds MAX_CLASSES the array would be too small.
 */
BUILD_ASSERT(TEST_NUM_BINS <= TEST_NUM_CLASSES,
	     "PROBS_DISTRIBUTION_BIN_NUM > MAX_CLASSES: enlarge counts[] in test_metric_capture");

/**
 * @brief Per-metric snapshot slot captured by tests.
 *
 * The library hands back a pointer into internal state via
 * nrf_edgeai_obsv_metric_snapshot_t. Tests copy the counters into a local
 * buffer so later updates do not invalidate the data under verification.
 */
struct test_metric_capture {
	bool present;
	uint32_t metric_id;
	uint32_t version;
	uint16_t num_rows;
	uint16_t num_cols;
	uint32_t counts[TEST_NUM_CLASSES * TEST_NUM_CLASSES];
};

/**
 * @brief Captured snapshots for the two known metrics plus a count of
 *        visited entries.
 */
struct test_snapshots {
	uint32_t visited;
	struct test_metric_capture probs_distribution;
	struct test_metric_capture transition_matrix;
};

/**
 * @brief for_each_metric callback that copies snapshots into
 *        @p user (a @c struct test_snapshots pointer) keyed by metric_id.
 */
bool test_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user);

#endif /* TEST_OBSV_COMMON_H */
