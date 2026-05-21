/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "common.h"

static void copy_snapshot(struct test_metric_capture *dst,
			  const nrf_edgeai_obsv_metric_snapshot_t *snap)
{
	dst->present = true;
	dst->metric_id = snap->metric_id;
	dst->version = snap->version;
	dst->num_rows = snap->num_rows;
	dst->num_cols = snap->num_cols;

	const size_t elems = (size_t)snap->num_rows * snap->num_cols;

	/* ARRAY_SIZE(dst->counts) sized to fit the worst-case
	 * (CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES * CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES) matrix.
	 * Anything else would signal a metric growing beyond what the test
	 * capture struct was sized for.
	 */
	zassert_true(elems <= ARRAY_SIZE(dst->counts), "snapshot too large for capture buffer: %zu",
		     elems);

	memcpy(dst->counts, snap->counts, elems * sizeof(dst->counts[0]));
}

bool test_capture_cb(const nrf_edgeai_obsv_metric_snapshot_t *snap, void *user)
{
	struct test_snapshots *snaps = user;

	snaps->visited++;

	switch (snap->metric_id) {
	case NRF_EDGEAI_OBSV_METRIC_ID_PROBS_DISTRIBUTION:
		copy_snapshot(&snaps->probs_distribution, snap);
		break;
	case NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX:
		copy_snapshot(&snaps->transition_matrix, snap);
		break;
	default:
		/* Ignore unknown metric ids so the tests can focus on the
		 * two metrics they actually exercise.
		 */
		break;
	}

	return true;
}
