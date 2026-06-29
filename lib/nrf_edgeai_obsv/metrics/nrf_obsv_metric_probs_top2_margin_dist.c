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

#define METRIC_PROBS_TOP2_MARGIN_DIST_VERSION 1

/*
 * Probs Top-2 Margin Distribution measures how decisive each prediction is. For
 * each inference it computes the margin between the two largest class
 * probabilities,
 *
 *   margin = p_top1 - p_top2,
 *
 * and accumulates it into a histogram. Because the probabilities are sorted and
 * non-negative, the margin lies in [0, 1] with no normalization needed: a
 * one-hot vector gives margin 1 (decisive), while two equally likely classes
 * give margin ~0 (ambiguous). A low margin flags ambiguous predictions even
 * when the dominant probability itself is high.
 *
 * Like the entropy distribution metric, the margin is a single scalar per
 * inference, so the histogram is one row of bin_num bins (1 x bin_num). The bins
 * are always uniform over [0, 1], so no edges are stored and values are binned in
 * O(1) via _dist_uniform_bin().
 *
 * Storage layout (mirrors the entropy distribution metric):
 *
 *   offset 0 : _nrf_obsv_pmd_hdr_t           (4 bytes)
 *   offset 4 : uint32_t counts[bin_num]
 *
 * (hdr + 1) steps past the header to the first counter.
 */

static inline uint32_t *pmd_counts(const _nrf_obsv_pmd_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

/* Margin between the two largest entries of @p p_probs, in [0, 1]. */
static float top2_margin(const float *p_probs, uint16_t n)
{
	float top1 = 0.0f;
	float top2 = 0.0f;

	for (uint16_t i = 0; i < n; i++) {
		if (p_probs[i] > top1) {
			top2 = top1;
			top1 = p_probs[i];
		} else if (p_probs[i] > top2) {
			top2 = p_probs[i];
		}
	}

	return top1 - top2;
}

static void pmd_clear(void *priv)
{
	const _nrf_obsv_pmd_hdr_t *hdr = priv;

	memset(pmd_counts(hdr), 0, sizeof(uint32_t) * hdr->bin_num);
}

static void pmd_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	pmd_clear(priv);
}

static void pmd_update(const float *p_probs, uint16_t n, void *priv)
{
	const _nrf_obsv_pmd_hdr_t *hdr = priv;
	uint32_t *counts = pmd_counts(hdr);

	assert(n <= hdr->num_classes);

	float margin = top2_margin(p_probs, n);

	counts[_dist_uniform_bin(hdr->bin_num, margin)]++;
}

static void pmd_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_pmd_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_PROBS_TOP2_MARGIN_DIST;
	out->version = METRIC_PROBS_TOP2_MARGIN_DIST_VERSION;
	out->num_rows = 1;
	out->num_cols = hdr->bin_num;
	out->counts = pmd_counts(hdr);
}

void nrf_edgeai_obsv_metric_pmd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_pmd_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_PROBS_TOP2_MARGIN_DIST_BIN_NUM;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = pmd_init,
		.update   = pmd_update,
		.clear    = pmd_clear,
		.finalize = NULL,
		.snapshot = pmd_snapshot,
		.source   = NRF_EDGEAI_OBSV_SOURCE_PROBS,
		.priv     = buf,
	};
}
