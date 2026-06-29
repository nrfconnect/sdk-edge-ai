/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define METRIC_PREDICTION_SWITCHING_RATE_VERSION 1

/* Sentinel stored in prev when no inference has been received. Class indices are
 * in [0, num_classes). At the maximum num_classes of 65535 (UINT16_MAX), valid
 * indices are [0, 65534], so 0xFFFF = 65535 is always outside the valid range.
 */
#define NO_PREV_CLASS 0xFFFFU

/*
 * Prediction Switching Rate tracks temporal instability: how often the dominant
 * class (argmax of the probability vector) changes between consecutive
 * inferences. High switching rates indicate instability or noisy inputs.
 *
 * Two raw counters are exported as a 1 x 2 row; the rate is derived off-device:
 *
 *   SwitchRate = switches / comparisons
 *              = (1 / (N - 1)) * sum_t I(y_t != y_{t-1})
 *
 * where comparisons == N - 1 (consecutive pairs seen since the last reset) and
 * switches == count of pairs whose dominant class differed.
 *
 * Storage layout:
 *
 *   offset 0 : _nrf_obsv_psr_hdr_t       (4 bytes)
 *   offset 4 : uint32_t counts[2]        (switches, comparisons)
 *
 * (hdr + 1) steps past the header to the first counter.
 */

#define PSR_IDX_SWITCHES    0U
#define PSR_IDX_COMPARISONS 1U
#define PSR_NUM_COUNTS      2U

static inline uint32_t *psr_counts(const _nrf_obsv_psr_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

static void psr_clear(void *priv)
{
	_nrf_obsv_psr_hdr_t *hdr = priv;

	memset(psr_counts(hdr), 0, sizeof(uint32_t) * PSR_NUM_COUNTS);
	hdr->prev = NO_PREV_CLASS;
}

static void psr_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	psr_clear(priv);
}

static void psr_update(const float *p_probs, uint16_t n, void *priv)
{
	_nrf_obsv_psr_hdr_t *hdr = priv;
	uint32_t *counts = psr_counts(hdr);

	assert(n <= hdr->num_classes);
	uint16_t cls = 0;
	float max_prob = p_probs[0];

	for (uint16_t i = 1; i < n; i++) {
		if (p_probs[i] > max_prob) {
			max_prob = p_probs[i];
			cls = i;
		}
	}

	if (hdr->prev != NO_PREV_CLASS) {
		counts[PSR_IDX_COMPARISONS]++;
		if (cls != hdr->prev) {
			counts[PSR_IDX_SWITCHES]++;
		}
	}

	hdr->prev = cls;
}

static void psr_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_psr_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_PREDICTION_SWITCHING_RATE;
	out->version = METRIC_PREDICTION_SWITCHING_RATE_VERSION;
	out->num_rows = 1;
	out->num_cols = PSR_NUM_COUNTS;
	out->counts = psr_counts(hdr);
}

void nrf_edgeai_obsv_metric_psr_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_psr_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->prev = NO_PREV_CLASS;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = psr_init,
		.update   = psr_update,
		.clear    = psr_clear,
		.finalize = NULL,
		.snapshot = psr_snapshot,
		.source   = NRF_EDGEAI_OBSV_SOURCE_PROBS,
		.priv     = buf,
	};
}
