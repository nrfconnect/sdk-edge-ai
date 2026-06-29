/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "nrf_obsv_dist_binning.h"

#define METRIC_PROBS_ENTROPY_DIST_VERSION 1

/*
 * Probs Entropy Distribution measures prediction uncertainty. For each inference
 * it computes the Shannon entropy of the output probability vector,
 *
 *   H(p) = -sum_i p_i * ln(p_i),
 *
 * normalizes it to [0, 1] by the maximum entropy ln(N) (uniform distribution
 * over N classes), and accumulates the result into a histogram. High entropy
 * (a flat distribution) flags uncertain predictions or out-of-distribution
 * inputs; low entropy (one dominant class) flags confident predictions.
 *
 * Unlike the per-class probability distribution metric, entropy is a single
 * scalar per inference, so the histogram is one row of bin_num bins (1 x bin_num).
 * The bins are always uniform over [0, 1], so no edges are stored and values are
 * binned in O(1) via _dist_uniform_bin().
 *
 * Storage layout:
 *
 *   offset 0 : _nrf_obsv_ped_hdr_t            (4 bytes)
 *   offset 4 : uint32_t counts[bin_num]
 *
 * (hdr + 1) steps past the header to the first counter.
 */

static inline uint32_t *ped_counts(const _nrf_obsv_ped_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

/* Normalized Shannon entropy of @p p_probs in [0, 1]; 0 when n < 2. */
static float normalized_entropy(const float *p_probs, uint16_t n)
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

	float h_norm = h / logf((float)n);

	h_norm = _clip01(h_norm);

	return h_norm;
}

static void ped_clear(void *priv)
{
	const _nrf_obsv_ped_hdr_t *hdr = priv;

	memset(ped_counts(hdr), 0, sizeof(uint32_t) * hdr->bin_num);
}

static void ped_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	ped_clear(priv);
}

static void ped_update(const float *p_probs, uint16_t n, void *priv)
{
	const _nrf_obsv_ped_hdr_t *hdr = priv;
	uint32_t *counts = ped_counts(hdr);

	assert(n <= hdr->num_classes);

	float h_norm = normalized_entropy(p_probs, n);

	counts[_dist_uniform_bin(hdr->bin_num, h_norm)]++;
}

static void ped_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_ped_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_PROBS_ENTROPY_DIST;
	out->version = METRIC_PROBS_ENTROPY_DIST_VERSION;
	out->num_rows = 1;
	out->num_cols = hdr->bin_num;
	out->counts = ped_counts(hdr);
}

void nrf_edgeai_obsv_metric_ped_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_ped_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_PROBS_ENTROPY_DIST_BIN_NUM;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = ped_init,
		.update   = ped_update,
		.clear    = ped_clear,
		.finalize = NULL,
		.snapshot = ped_snapshot,
		.source   = NRF_EDGEAI_OBSV_SOURCE_PROBS,
		.priv     = buf,
	};
}
