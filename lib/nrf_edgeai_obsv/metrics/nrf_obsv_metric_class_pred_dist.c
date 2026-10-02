/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "nrf_obsv_dist_binning.h"

#define METRIC_CLASS_PRED_DIST_VERSION 1

/* Sentinel stored in cur_class when no streak is active. Class indices are in
 * [0, num_classes); at the maximum num_classes of 65535 (UINT16_MAX) the valid
 * indices are [0, 65534], so 0xFFFF is always outside the range.
 */
#define NO_CUR_CLASS 0xFFFFU

/*
 * Class Predictions Distribution gives a per-class picture of the model's output
 * in one 2*num_classes x bin_num matrix:
 *
 *   rows [0, num_classes)          probability distribution: each class's
 *                                  predicted probability binned uniformly over
 *                                  [0, 1]. Every inference adds one sample to
 *                                  every class row, so each row sums to the
 *                                  inference count.
 *   rows [num_classes, 2*num_classes)  streak distribution: per class, a
 *                                  histogram of how many consecutive inferences
 *                                  the dominant class (argmax) stays that class,
 *                                  binned uniformly over [1, top]. Recorded only
 *                                  when a streak ends; these rows do not sum to
 *                                  the inference count.
 *
 * One bin_num is shared by both row groups.
 *
 * Storage layout:
 *
 *   offset 0  : _nrf_obsv_cpd_hdr_t                          (16 bytes)
 *   offset 16 : uint32_t counts[2 * num_classes][bin_num]
 *               (first num_classes rows = probs, next num_classes = streaks)
 *
 * (hdr + 1) steps past the header to the first counter.
 */

static inline uint32_t *cpd_counts(const _nrf_obsv_cpd_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

/* Base of the streak row group (rows [num_classes, 2*num_classes)). */
static inline uint32_t *cpd_streak(const _nrf_obsv_cpd_hdr_t *hdr)
{
	return cpd_counts(hdr) + (size_t)hdr->num_classes * hdr->bin_num;
}

/* Record a completed streak of length @len (in [1, top]) for class @cls into the
 * streak row group: bin the length uniformly over [1, top] (len == top or longer
 * lands in the top bin) and bump the class's histogram row.
 */
static void cpd_record_streak(const _nrf_obsv_cpd_hdr_t *hdr, uint16_t cls, uint8_t len)
{
	uint32_t *streak = cpd_streak(hdr);
	const uint32_t top = hdr->cfg[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOP];
	float x = (top > 1U) ? ((float)(len - 1U) / (float)(top - 1U)) : 0.0f;
	uint8_t bin = _dist_uniform_bin(hdr->bin_num, x);

	streak[(size_t)cls * hdr->bin_num + bin]++;
}

static void cpd_clear(void *priv)
{
	_nrf_obsv_cpd_hdr_t *hdr = priv;

	memset(cpd_counts(hdr), 0,
	       sizeof(uint32_t) * (size_t)NRF_EDGEAI_OBSV_CPD_ROW_GROUPS * hdr->num_classes *
		       hdr->bin_num);
	hdr->cur_class = NO_CUR_CLASS;
	hdr->cur_len = 0;
	hdr->cur_miss = 0;
}

static void cpd_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	cpd_clear(priv);
}

static void cpd_update(const float *p_probs, uint16_t n, void *priv)
{
	_nrf_obsv_cpd_hdr_t *hdr = priv;
	uint32_t *prob = cpd_counts(hdr);

	assert(n <= hdr->num_classes);

	/* Probability distribution: bin every class's probability, and find argmax. */
	uint16_t cls = 0;
	float max_prob = p_probs[0];

	for (uint16_t i = 0; i < n; i++) {
		prob[(size_t)i * hdr->bin_num + _dist_uniform_bin(hdr->bin_num, p_probs[i])]++;
		if (p_probs[i] > max_prob) {
			max_prob = p_probs[i];
			cls = i;
		}
	}

	/* Streak distribution: track the dominant-class run with flicker tolerance. */
	if (hdr->cur_class == NO_CUR_CLASS) {
		/* Start the first streak. */
		hdr->cur_class = cls;
		hdr->cur_len = 1;
		hdr->cur_miss = 0;
	} else if (cls == hdr->cur_class) {
		/* Extend: count this matched frame (capped at top) and refill tolerance. */
		if (hdr->cur_len < hdr->cfg[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOP]) {
			hdr->cur_len++;
		}
		hdr->cur_miss = 0;
	} else if (hdr->cur_miss >= hdr->cfg[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOL]) {
		/* Tolerance exhausted: the streak ends here; the breaking frame starts
		 * a new streak of its own class.
		 */
		cpd_record_streak(hdr, hdr->cur_class, hdr->cur_len);
		hdr->cur_class = cls;
		hdr->cur_len = 1;
		hdr->cur_miss = 0;
	} else {
		/* Bridge a tolerated flicker: consume one tolerance unit; the frame is
		 * not counted into the streak length.
		 */
		hdr->cur_miss++;
	}
}

static void cpd_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_cpd_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_CLASS_PRED_DIST;
	out->version = METRIC_CLASS_PRED_DIST_VERSION;
	out->num_rows = (uint16_t)(NRF_EDGEAI_OBSV_CPD_ROW_GROUPS * hdr->num_classes);
	out->num_cols = hdr->bin_num;
	out->counts = cpd_counts(hdr);
	out->config_rows = 1;
	out->config_cols = NRF_EDGEAI_OBSV_CPD_CFG_COUNT;
	out->config = hdr->cfg;
}

void nrf_edgeai_obsv_metric_cpd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_cpd_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM;
	hdr->cfg[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOP] =
		CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOP_BIN;
	hdr->cfg[NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOL] =
		CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOL;
	hdr->cur_class = NO_CUR_CLASS;
	hdr->cur_len = 0;
	hdr->cur_miss = 0;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init = cpd_init,
		.update = cpd_update,
		.clear = cpd_clear,
		.finalize = NULL,
		.snapshot = cpd_snapshot,
		.source = NRF_EDGEAI_OBSV_SOURCE_PROBS,
		.priv = buf,
	};
}
