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

#define METRIC_MEL_ENERGY_DESC_VERSION 1

/* Configured percentiles must define a non-empty range. */
_Static_assert(CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P99_MILLI >
		       CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P01_MILLI,
	       "SCALE_P99_MILLI must be greater than SCALE_P01_MILLI");

/* Row indices in the descriptor snapshot (num_rows == NRF_EDGEAI_OBSV_MED_NUM_ROWS). */
#define MED_ROW_MEAN_ENERGY   0U
#define MED_ROW_MAX_ENERGY    1U
#define MED_ROW_DYNAMIC_RANGE 2U
#define MED_ROW_FLOOR_RATIO   3U

/*
 * Mel Energy Descriptor folds per-frame summary statistics of the mel feature
 * vector into one histogram per statistic. Each feature value is first
 * normalized into [0, 1] against the configured percentile range [p01, p99]:
 *
 *   x_norm[b] = clamp((x[b] - p01) / (p99 - p01), 0, 1)
 *
 * so a single uniform [0, 1] binning applies to every row, and the bins are
 * absolute and comparable across devices/windows (the percentiles are measured
 * offline on a representative dataset and supplied via Kconfig):
 *
 *   row 0 mean_energy   = mean_b x_norm[b]
 *   row 1 max_energy    = max_b  x_norm[b]
 *   row 2 dynamic_range = q95_b x_norm - q05_b x_norm
 *   row 3 floor_ratio   = mean_b I(x[b] <= 0)     (raw silence floor, scale-free)
 *
 * Every row is uniform over [0, 1], so no edges are stored and values are binned
 * in O(1) via _dist_uniform_bin().
 *
 * Storage layout:
 *
 *   offset 0  : _nrf_obsv_med_hdr_t                  (12 bytes)
 *   offset 12 : uint32_t counts[NUM_ROWS * bin_num]  (row-major)
 */

static inline uint32_t *med_counts(const _nrf_obsv_med_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

/* Ascending insertion sort of @n values in place. */
static void sort_inplace(float *a, uint16_t n)
{
	for (uint16_t i = 1; i < n; i++) {
		float v = a[i];
		uint16_t j = i;

		while (j > 0 && a[j - 1] > v) {
			a[j] = a[j - 1];
			j--;
		}
		a[j] = v;
	}
}

/* Nearest-rank index for quantile fraction @q over @n sorted samples. */
static uint16_t rank_index(float q, uint16_t n)
{
	return (uint16_t)(q * (float)(n - 1) + 0.5f);
}

static void med_clear(void *priv)
{
	const _nrf_obsv_med_hdr_t *hdr = priv;

	memset(med_counts(hdr), 0,
	       sizeof(uint32_t) * NRF_EDGEAI_OBSV_MED_NUM_ROWS * hdr->bin_num);
}

static void med_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	med_clear(priv);
}

static void med_update(const float *p_feats, uint16_t n, void *priv)
{
	const _nrf_obsv_med_hdr_t *hdr = priv;
	uint32_t *counts = med_counts(hdr);
	const uint8_t bins = hdr->bin_num;

	assert(n <= hdr->num_features);
	assert(n <= CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES);

	if (n == 0) {
		return;
	}

	/* Hard cap so a misconfigured n_features can never overflow scratch[] in
	 * release builds where the asserts above are compiled out.
	 */
	if (n > CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES) {
		n = CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES;
	}

	const float range = hdr->scale_max - hdr->scale_min;
	float scratch[CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES];

	float sum_n = 0.0f;
	float max_n = 0.0f;
	uint16_t floor_cnt = 0;

	for (uint16_t b = 0; b < n; b++) {
		float v = p_feats[b];

		if (v <= 0.0f) {
			floor_cnt++;
		}

		float vn = (v - hdr->scale_min) / range;

		vn = _clip01(vn);

		sum_n += vn;
		if (vn > max_n) {
			max_n = vn;
		}
		scratch[b] = vn;
	}

	sort_inplace(scratch, n);

	float mean_n = sum_n / (float)n;
	float dyn_range = scratch[rank_index(0.95f, n)] - scratch[rank_index(0.05f, n)];
	float floor_ratio = (float)floor_cnt / (float)n;

	counts[MED_ROW_MEAN_ENERGY * bins + _dist_uniform_bin(bins, mean_n)]++;
	counts[MED_ROW_MAX_ENERGY * bins + _dist_uniform_bin(bins, max_n)]++;
	counts[MED_ROW_DYNAMIC_RANGE * bins + _dist_uniform_bin(bins, dyn_range)]++;
	counts[MED_ROW_FLOOR_RATIO * bins + _dist_uniform_bin(bins, floor_ratio)]++;
}

static void med_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_med_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_MEL_ENERGY_DESC;
	out->version = METRIC_MEL_ENERGY_DESC_VERSION;
	out->num_rows = NRF_EDGEAI_OBSV_MED_NUM_ROWS;
	out->num_cols = hdr->bin_num;
	out->counts = med_counts(hdr);
}

void nrf_edgeai_obsv_metric_med_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_features)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);
	assert(n_features <= CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES);

	_nrf_obsv_med_hdr_t *hdr = buf;

	hdr->num_features = n_features;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM;
	hdr->scale_min = (float)CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P01_MILLI / 1000.0f;
	hdr->scale_max = (float)CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P99_MILLI / 1000.0f;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = med_init,
		.update   = med_update,
		.clear    = med_clear,
		.finalize = NULL,
		.snapshot = med_snapshot,
		.source   = NRF_EDGEAI_OBSV_SOURCE_FEATURES,
		.priv     = buf,
	};
}
