/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "nrf_obsv_dist_binning.h"

#define METRIC_MODEL_CERTAINTY_DESC_VERSION 1

/*
 * Model Certainty Descriptor summarizes, per inference, how certain and how
 * temporally stable the model's predictions are.
 *
 * For each inference it derives, in one pass over the probability vector:
 *   - the normalized Shannon entropy H(p)/ln(N) in [0, 1] (uncertainty),
 *   - the top-2 margin p_top1 - p_top2 in [0, 1] (decisiveness),
 *   - the dominant class (argmax) and whether it changed vs the previous
 *     inference (temporal stability).
 *
 * Matrix layout (row-major, NRF_EDGEAI_OBSV_MCD_NUM_ROWS x bin_num):
 *   row 0 (ENTROPY)   : histogram of normalized entropy over bin_num [0, 1] bins
 *   row 1 (MARGIN)    : histogram of the top-2 margin over bin_num [0, 1] bins
 *   row 2 (STABILITY) : counters, the rest of the row zero-padded:
 *                         col 0 switches           (argmax changed vs prev)
 *                         col 1 comparisons        (consecutive pairs seen)
 *                         col 2 majority_frames     (inferences with p_top1 > 0.5)
 *                         col 3 confident_switches  (switches into a p_top1 > 0.5 winner)
 *
 *
 * Both histogram rows share one bin count (bin_num), always uniform over [0, 1],
 * so no edges are stored and values are binned in O(1) via _dist_uniform_bin().
 *
 * Storage layout:
 *
 *   offset 0 : _nrf_obsv_mcd_hdr_t                        (8 bytes)
 *   offset 8 : uint32_t counts[NRF_EDGEAI_OBSV_MCD_NUM_ROWS * bin_num]
 *
 * (hdr + 1) steps past the header to the first counter.
 */

#define MCD_ROW_ENTROPY	  0U
#define MCD_ROW_MARGIN	  1U
#define MCD_ROW_STABILITY 2U

#define MCD_STAB_SWITCHES	  0U
#define MCD_STAB_COMPARISONS	  1U
#define MCD_STAB_MAJORITY_FRAMES  2U
#define MCD_STAB_CONFIDENT_SWITCH 3U

/* Absolute-majority boundary: a winner above this holds more than half the mass. */
#define MCD_MAJORITY_THRESHOLD 0.5f

/* Sentinel stored in prev when no inference has been received. Class indices are
 * in [0, num_classes). At the maximum num_classes of 65535 (UINT16_MAX), valid
 * indices are [0, 65534], so 0xFFFF = 65535 is always outside the valid range.
 */
#define MCD_NO_PREV_CLASS 0xFFFFU

static inline uint32_t *mcd_counts(const _nrf_obsv_mcd_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

static inline uint32_t *mcd_row(const _nrf_obsv_mcd_hdr_t *hdr, uint8_t row)
{
	return mcd_counts(hdr) + (size_t)row * hdr->bin_num;
}

/* Normalized Shannon entropy of @p p_probs in [0, 1]; 0 when n < 2. */
static float mcd_normalized_entropy(const float *p_probs, uint16_t n)
{
	if (n < 2) {
		return 0.0f;
	}

	float h = 0.0f;

	for (uint16_t i = 0; i < n; i++) {
		if (p_probs[i] > 0.0f) {
			h -= p_probs[i] * logf(p_probs[i]);
		}
	}

	return _clip01(h / logf((float)n));
}

static void mcd_clear(void *priv)
{
	_nrf_obsv_mcd_hdr_t *hdr = priv;

	memset(mcd_counts(hdr), 0,
	       sizeof(uint32_t) * (size_t)NRF_EDGEAI_OBSV_MCD_NUM_ROWS * hdr->bin_num);
	hdr->prev = MCD_NO_PREV_CLASS;
}

static void mcd_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	mcd_clear(priv);
}

static void mcd_update(const float *p_probs, uint16_t n, void *priv)
{
	_nrf_obsv_mcd_hdr_t *hdr = priv;

	assert(n <= hdr->num_classes);

	/* Dominant class and the two largest probabilities, in one pass. */
	uint16_t cls = 0;
	float top1 = 0.0f;
	float top2 = 0.0f;

	for (uint16_t i = 0; i < n; i++) {
		if (p_probs[i] > top1) {
			top2 = top1;
			top1 = p_probs[i];
			cls = i;
		} else if (p_probs[i] > top2) {
			top2 = p_probs[i];
		}
	}

	float h_norm = mcd_normalized_entropy(p_probs, n);
	float margin = top1 - top2;

	mcd_row(hdr, MCD_ROW_ENTROPY)[_dist_uniform_bin(hdr->bin_num, h_norm)]++;
	mcd_row(hdr, MCD_ROW_MARGIN)[_dist_uniform_bin(hdr->bin_num, margin)]++;

	uint32_t *stab = mcd_row(hdr, MCD_ROW_STABILITY);
	bool majority = top1 > MCD_MAJORITY_THRESHOLD;

	if (hdr->prev != MCD_NO_PREV_CLASS) {
		stab[MCD_STAB_COMPARISONS]++;
		if (cls != hdr->prev) {
			stab[MCD_STAB_SWITCHES]++;
			if (majority) {
				stab[MCD_STAB_CONFIDENT_SWITCH]++;
			}
		}
	}

	if (majority) {
		stab[MCD_STAB_MAJORITY_FRAMES]++;
	}

	hdr->prev = cls;
}

static void mcd_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_mcd_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_MODEL_CERTAINTY_DESC;
	out->version = METRIC_MODEL_CERTAINTY_DESC_VERSION;
	out->num_rows = NRF_EDGEAI_OBSV_MCD_NUM_ROWS;
	out->num_cols = hdr->bin_num;
	out->counts = mcd_counts(hdr);
}

void nrf_edgeai_obsv_metric_mcd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_mcd_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM;
	hdr->prev = MCD_NO_PREV_CLASS;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init = mcd_init,
		.update = mcd_update,
		.clear = mcd_clear,
		.finalize = NULL,
		.snapshot = mcd_snapshot,
		.source = NRF_EDGEAI_OBSV_SOURCE_PROBS,
		.priv = buf,
	};
}
