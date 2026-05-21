/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define METRIC_PROBS_DISTRIBUTION_VERSION 1

/*
 * Storage layout:
 *
 *   offset 0 : _nrf_obsv_pd_hdr_t              (4 bytes)
 *   offset 4 : float inner_edges[bin_num - 1]
 *   offset 4 + (bin_num - 1) * 4 : uint32_t counts[num_classes][bin_num]
 *
 * (hdr + 1) steps past the header to the first inner edge.
 * (edges + bin_num - 1) steps past the edge array to the first counter.
 */

static inline float *pd_edges(const _nrf_obsv_pd_hdr_t *hdr)
{
	return (float *)(hdr + 1);
}

static inline uint32_t *pd_counts(const _nrf_obsv_pd_hdr_t *hdr)
{
	return (uint32_t *)(pd_edges(hdr) + hdr->bin_num - 1);
}

static void compute_uniform_edges(float *edges, uint8_t bin_num)
{
	const float step = 1.0f / bin_num;

	for (int i = 0; i < bin_num - 1; i++) {
		edges[i] = (i + 1) * step;
	}
}

static uint8_t find_bin(const float *edges, uint8_t bin_num, float val)
{
	for (uint8_t b = 0; b < bin_num - 1; b++) {
		if (val < edges[b]) {
			return b;
		}
	}
	return bin_num - 1;
}

static void pd_init(const void *p_cfg, void *priv)
{
	const _nrf_obsv_pd_hdr_t *hdr = priv;
	float *edges = pd_edges(hdr);
	uint32_t *counts = pd_counts(hdr);

	memset(counts, 0, sizeof(uint32_t) * (size_t)hdr->num_classes * hdr->bin_num);

	if (p_cfg != NULL) {
		const nrf_obsv_probs_dist_cfg_t *c = p_cfg;

		memcpy(edges, c->bin_edges, sizeof(float) * ((size_t)hdr->bin_num - 1));
		return;
	}

	compute_uniform_edges(edges, hdr->bin_num);
}

static void pd_update(const float *p_probs, uint16_t n, void *priv)
{
	const _nrf_obsv_pd_hdr_t *hdr = priv;
	const float *edges = pd_edges(hdr);
	uint32_t *counts = pd_counts(hdr);

	assert(n <= hdr->num_classes);

	for (uint16_t i = 0; i < n; i++) {
		counts[(size_t)i * hdr->bin_num + find_bin(edges, hdr->bin_num, p_probs[i])]++;
	}
}

static void pd_clear(void *priv)
{
	const _nrf_obsv_pd_hdr_t *hdr = priv;

	memset(pd_counts(hdr), 0, sizeof(uint32_t) * (size_t)hdr->num_classes * hdr->bin_num);
}

static void pd_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_pd_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_PROBS_DISTRIBUTION;
	out->version = METRIC_PROBS_DISTRIBUTION_VERSION;
	out->num_rows = hdr->num_classes;
	out->num_cols = hdr->bin_num;
	out->counts = pd_counts(hdr);
}

void nrf_edgeai_obsv_metric_pd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				      uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_pd_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = pd_init,
		.update   = pd_update,
		.clear    = pd_clear,
		.finalize = NULL,
		.snapshot = pd_snapshot,
		.priv     = buf,
	};
}
