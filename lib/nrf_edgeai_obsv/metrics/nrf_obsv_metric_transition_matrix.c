/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#define METRIC_TRANSITION_MATRIX_VERSION 1

/* Sentinel stored in prev when no inference has been received. Class indices are
 * in [0, num_classes). At the maximum num_classes of 65535 (UINT16_MAX), valid
 * indices are [0, 65534], so 0xFFFF = 65535 is always outside the valid range.
 */
#define NO_PREV_CLASS 0xFFFFU

/*
 * Storage layout:
 *
 *   offset 0 : _nrf_obsv_tm_hdr_t                    (4 bytes)
 *   offset 4 : uint32_t matrix[num_classes][num_classes]
 *
 * (hdr + 1) steps past the header to the first matrix element.
 */

static inline uint32_t *tm_matrix(const _nrf_obsv_tm_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

static void tm_clear(void *priv)
{
	_nrf_obsv_tm_hdr_t *hdr = priv;
	uint32_t *matrix = tm_matrix(hdr);

	memset(matrix, 0, sizeof(uint32_t) * (size_t)hdr->num_classes * hdr->num_classes);
	hdr->prev = NO_PREV_CLASS;
}

static void tm_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	tm_clear(priv);
}

static void tm_update(const float *p_probs, uint16_t n, void *priv)
{
	_nrf_obsv_tm_hdr_t *hdr = priv;
	uint32_t *matrix = tm_matrix(hdr);

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
		matrix[(size_t)hdr->prev * hdr->num_classes + cls]++;
	}

	hdr->prev = cls;
}

static void tm_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_tm_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX;
	out->version = METRIC_TRANSITION_MATRIX_VERSION;
	out->num_rows = hdr->num_classes;
	out->num_cols = hdr->num_classes;
	out->counts = tm_matrix(hdr);
}

void nrf_edgeai_obsv_metric_tm_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				      uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_tm_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->prev = NO_PREV_CLASS;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = tm_init,
		.update   = tm_update,
		.clear    = tm_clear,
		.finalize = NULL,
		.snapshot = tm_snapshot,
		.source   = NRF_EDGEAI_OBSV_SOURCE_PROBS,
		.priv     = buf,
	};
}
