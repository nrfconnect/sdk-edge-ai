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

#define METRIC_MEL_SPECTRAL_DESC_VERSION 1

/* Avoids division by zero on silent frames and log(0) in flatness/entropy. */
#define MSD_EPS 1e-6f

/* Row indices in the descriptor snapshot (num_rows == NRF_EDGEAI_OBSV_MSD_NUM_ROWS). */
#define MSD_ROW_LOW_RATIO     0U
#define MSD_ROW_MID_RATIO     1U
#define MSD_ROW_HIGH_RATIO    2U
#define MSD_ROW_CENTROID      3U
#define MSD_ROW_SPREAD        4U
#define MSD_ROW_ENTROPY       5U
#define MSD_ROW_FLATNESS      6U
#define MSD_ROW_CONTRAST      7U

/*
 * Mel Spectral Descriptor folds per-frame spectral-shape statistics of the mel
 * feature vector into one histogram per statistic. Every statistic is
 * scale-invariant by construction (it divides by the total energy S or the mean),
 * so no amplitude calibration is needed; values are mapped to [0, 1] and binned
 * in O(1) via _dist_uniform_bin().
 *
 * Precondition: feature values are non-negative (they form an energy spectrum).
 * The metric clamps negatives to 0 defensively so that logf()/sqrtf() and the
 * p[b] = x[b]/S probabilities stay well-defined. With p[b] = x[b] / S and the
 * band split into thirds (b1 = n/3, b2 = 2n/3 by integer division):
 *
 *   row 0 low_ratio   = sum_{b<b1} x[b] / S
 *   row 1 mid_ratio   = sum_{b1<=b<b2} x[b] / S
 *   row 2 high_ratio  = sum_{b>=b2} x[b] / S
 *   row 3 centroid    = (sum_b b*p[b]) / (n - 1)
 *   row 4 spread      = sqrt(sum_b (b-centroid_raw)^2 * p[b]) / ((n - 1) / 2)
 *   row 5 entropy     = -sum_b p[b] ln p[b] / ln(n)
 *   row 6 flatness    = geomean_b(x[b]+eps) / (mean_b x[b] + eps)
 *   row 7 contrast    = ((max_b x - min_b x) / (mean_b x + eps)) / n
 *
 * Rows 3/4/7 are clamped to [0, 1] (their theoretical maxima — uniform spectrum,
 * two-point-mass spectrum, one-hot spectrum — are rarely reached).
 *
 * Storage layout:
 *
 *   offset 0 : _nrf_obsv_msd_hdr_t                  (4 bytes)
 *   offset 4 : uint32_t counts[NUM_ROWS * bin_num]  (row-major)
 */

static inline uint32_t *msd_counts(const _nrf_obsv_msd_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

static void msd_clear(void *priv)
{
	const _nrf_obsv_msd_hdr_t *hdr = priv;

	memset(msd_counts(hdr), 0,
	       sizeof(uint32_t) * NRF_EDGEAI_OBSV_MSD_NUM_ROWS * hdr->bin_num);
}

static void msd_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	msd_clear(priv);
}

static void msd_update(const float *p_feats, uint16_t n, void *priv)
{
	const _nrf_obsv_msd_hdr_t *hdr = priv;
	uint32_t *counts = msd_counts(hdr);
	const uint8_t bins = hdr->bin_num;

	assert(n <= hdr->num_features);

	if (n == 0) {
		return;
	}

	const uint16_t b1 = n / 3U;
	const uint16_t b2 = (2U * n) / 3U;

	/* Pass 1: energy, band sums, weighted index sum, geomean accumulator, extremes. */
	float total = 0.0f;
	float sum_bx = 0.0f;
	float sum_log = 0.0f;
	float low_s = 0.0f, mid_s = 0.0f, high_s = 0.0f;
	float fmax = (p_feats[0] < 0.0f) ? 0.0f : p_feats[0];
	float fmin = fmax;

	for (uint16_t b = 0; b < n; b++) {
		float v = p_feats[b];

		/* Spectrum energy is non-negative; clamp defensively so logf() and
		 * the p[b] = x[b]/S probabilities below stay well-defined.
		 */
		if (v < 0.0f) {
			v = 0.0f;
		}

		total += v;
		sum_bx += (float)b * v;
		sum_log += logf(v + MSD_EPS);

		if (v > fmax) {
			fmax = v;
		}
		if (v < fmin) {
			fmin = v;
		}

		if (b < b1) {
			low_s += v;
		} else if (b < b2) {
			mid_s += v;
		} else {
			high_s += v;
		}
	}

	const float inv_s = 1.0f / (total + MSD_EPS);
	const float centroid = sum_bx * inv_s; /* raw, in [0, n-1] */

	/* Pass 2: spread around the centroid and Shannon entropy of p[b]. */
	float var = 0.0f;
	float ent = 0.0f;

	for (uint16_t b = 0; b < n; b++) {
		float v = (p_feats[b] < 0.0f) ? 0.0f : p_feats[b];
		float p = v * inv_s;
		float d = (float)b - centroid;

		var += d * d * p;
		if (p > 0.0f) {
			ent -= p * logf(p);
		}
	}

	const float mean = total / (float)n;
	const float geomean = expf(sum_log / (float)n);
	const float span = (float)(n - 1);

	float low_ratio = _clip01(low_s * inv_s);
	float mid_ratio = _clip01(mid_s * inv_s);
	float high_ratio = _clip01(high_s * inv_s);
	float centroid_n = (n >= 2) ? _clip01(centroid / span) : 0.0f;
	float spread_n = (n >= 2) ? _clip01(sqrtf(var) / (span * 0.5f)) : 0.0f;
	float entropy_n = (n >= 2) ? _clip01(ent / logf((float)n)) : 0.0f;
	float flatness = _clip01(geomean / (mean + MSD_EPS));
	float contrast_n = _clip01(((fmax - fmin) / (mean + MSD_EPS)) / (float)n);

	counts[MSD_ROW_LOW_RATIO * bins + _dist_uniform_bin(bins, low_ratio)]++;
	counts[MSD_ROW_MID_RATIO * bins + _dist_uniform_bin(bins, mid_ratio)]++;
	counts[MSD_ROW_HIGH_RATIO * bins + _dist_uniform_bin(bins, high_ratio)]++;
	counts[MSD_ROW_CENTROID * bins + _dist_uniform_bin(bins, centroid_n)]++;
	counts[MSD_ROW_SPREAD * bins + _dist_uniform_bin(bins, spread_n)]++;
	counts[MSD_ROW_ENTROPY * bins + _dist_uniform_bin(bins, entropy_n)]++;
	counts[MSD_ROW_FLATNESS * bins + _dist_uniform_bin(bins, flatness)]++;
	counts[MSD_ROW_CONTRAST * bins + _dist_uniform_bin(bins, contrast_n)]++;
}

static void msd_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_msd_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_MEL_SPECTRAL_DESC;
	out->version = METRIC_MEL_SPECTRAL_DESC_VERSION;
	out->num_rows = NRF_EDGEAI_OBSV_MSD_NUM_ROWS;
	out->num_cols = hdr->bin_num;
	out->counts = msd_counts(hdr);
}

void nrf_edgeai_obsv_metric_msd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_features)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_msd_hdr_t *hdr = buf;

	hdr->num_features = n_features;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = msd_init,
		.update   = msd_update,
		.clear    = msd_clear,
		.finalize = NULL,
		.snapshot = msd_snapshot,
		.source   = NRF_EDGEAI_OBSV_SOURCE_FEATURES,
		.priv     = buf,
	};
}
