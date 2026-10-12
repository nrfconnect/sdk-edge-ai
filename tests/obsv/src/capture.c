/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "common.h"

/* Copy @n_rows rows starting at row @row_off of @snap into @dst. */
static void copy_rows(struct test_metric_capture *dst,
		      const nrf_edgeai_obsv_metric_snapshot_t *snap,
		      uint16_t row_off, uint16_t n_rows)
{
	dst->present = true;
	dst->metric_id = snap->metric_id;
	dst->version = snap->version;
	dst->num_rows = n_rows;
	dst->num_cols = snap->num_cols;
	dst->config_rows = (snap->config != NULL) ? snap->config_rows : 0;
	dst->config_cols = (snap->config != NULL) ? snap->config_cols : 0;
	if (snap->config != NULL) {
		const size_t cfg_elems = (size_t)snap->config_rows * snap->config_cols;

		zassert_true(cfg_elems <= ARRAY_SIZE(dst->config),
			     "config too large for capture buffer: %zu", cfg_elems);
		memcpy(dst->config, snap->config, cfg_elems * sizeof(dst->config[0]));
	}

	const size_t elems = (size_t)n_rows * snap->num_cols;

	/* ARRAY_SIZE(dst->counts) sized to fit the worst-case
	 * (CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES * CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES) matrix.
	 * Anything else would signal a metric growing beyond what the test
	 * capture struct was sized for.
	 */
	zassert_true(elems <= ARRAY_SIZE(dst->counts), "snapshot too large for capture buffer: %zu",
		     elems);

	memcpy(dst->counts, snap->counts + (size_t)row_off * snap->num_cols,
	       elems * sizeof(dst->counts[0]));
}

static void copy_snapshot(struct test_metric_capture *dst,
			  const nrf_edgeai_obsv_metric_snapshot_t *snap)
{
	copy_rows(dst, snap, 0, snap->num_rows);
}

bool test_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct test_snapshots *snaps = user;

	snaps->visited++;

	switch (snap->metric_id) {
	case NRF_EDGEAI_OBSV_METRIC_ID_CLASS_PRED_DIST: {
		/* 2*num_classes x bin_num: rows [0, N) probability distribution,
		 * rows [N, 2N) streak distribution. Split into two per-class views.
		 */
		uint16_t n = snap->num_rows / 2U;

		copy_rows(&snaps->probs_distribution, snap, 0, n);
		copy_rows(&snaps->class_streak, snap, n, n);
		break;
	}
	case NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX:
		copy_snapshot(&snaps->transition_matrix, snap);
		break;
	default:
		/* Ignore unknown metric ids so the tests can focus on the
		 * metrics they actually exercise.
		 */
		break;
	}

	return true;
}
