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

#define METRIC_CLASS_STREAK_DIST_VERSION 1

/* Sentinel stored in cur_class when no streak is active. Class indices are in
 * [0, num_classes); at the maximum num_classes of 65535 (UINT16_MAX) the valid
 * indices are [0, 65534], so 0xFFFF is always outside the range.
 */
#define NO_CUR_CLASS 0xFFFFU

/*
 * Storage layout:
 *
 *   offset 0  : _nrf_obsv_csd_hdr_t                     (12 bytes)
 *   offset 12 : uint32_t counts[num_classes][bin_num]
 *
 * (hdr + 1) steps past the header to the first counter.
 */

static inline uint32_t *csd_counts(const _nrf_obsv_csd_hdr_t *hdr)
{
	return (uint32_t *)(hdr + 1);
}

/* Record a completed streak of length @len (in [1, top]) for class @cls: bin the
 * length uniformly over [1, top] (len == top or longer lands in the top bin) and
 * bump the class's histogram row.
 */
static void csd_record(const _nrf_obsv_csd_hdr_t *hdr, uint16_t cls, uint8_t len)
{
	uint32_t *counts = csd_counts(hdr);
	float x = (hdr->top > 1U) ? ((float)(len - 1U) / (float)(hdr->top - 1U)) : 0.0f;
	uint8_t bin = _dist_uniform_bin(hdr->bin_num, x);

	counts[(size_t)cls * hdr->bin_num + bin]++;
}

static void csd_clear(void *priv)
{
	_nrf_obsv_csd_hdr_t *hdr = priv;

	memset(csd_counts(hdr), 0, sizeof(uint32_t) * (size_t)hdr->num_classes * hdr->bin_num);
	hdr->cur_class = NO_CUR_CLASS;
	hdr->cur_len = 0;
	hdr->cur_miss = 0;
}

static void csd_init(const void *p_cfg, void *priv)
{
	(void)p_cfg;

	csd_clear(priv);
}

static void csd_update(const float *p_probs, uint16_t n, void *priv)
{
	_nrf_obsv_csd_hdr_t *hdr = priv;

	assert(n <= hdr->num_classes);

	uint16_t cls = 0;
	float max_prob = p_probs[0];

	for (uint16_t i = 1; i < n; i++) {
		if (p_probs[i] > max_prob) {
			max_prob = p_probs[i];
			cls = i;
		}
	}

	if (hdr->cur_class == NO_CUR_CLASS) {
		/* Start the first streak. */
		hdr->cur_class = cls;
		hdr->cur_len = 1;
		hdr->cur_miss = 0;
	} else if (cls == hdr->cur_class) {
		/* Extend: count this matched frame (capped at top) and refill tolerance. */
		if (hdr->cur_len < hdr->top) {
			hdr->cur_len++;
		}
		hdr->cur_miss = 0;
	} else if (hdr->cur_miss >= hdr->tolerance) {
		/* Tolerance exhausted: the streak ends here; the breaking frame starts
		 * a new streak of its own class.
		 */
		csd_record(hdr, hdr->cur_class, hdr->cur_len);
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

static void csd_snapshot(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv)
{
	const _nrf_obsv_csd_hdr_t *hdr = priv;

	out->metric_id = NRF_EDGEAI_OBSV_METRIC_ID_CLASS_STREAK_DIST;
	out->version = METRIC_CLASS_STREAK_DIST_VERSION;
	out->num_rows = hdr->num_classes;
	out->num_cols = hdr->bin_num;
	out->counts = csd_counts(hdr);
}

void nrf_edgeai_obsv_metric_csd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes)
{
	assert((uintptr_t)buf % sizeof(uint32_t) == 0);

	_nrf_obsv_csd_hdr_t *hdr = buf;

	hdr->num_classes = n_classes;
	hdr->bin_num = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_BIN_NUM;
	hdr->top = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOP;
	hdr->tolerance = (uint8_t)CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOLERANCE;
	hdr->cur_class = NO_CUR_CLASS;
	hdr->cur_len = 0;
	hdr->cur_miss = 0;

	*metric = (nrf_edgeai_obsv_metric_t){
		.init     = csd_init,
		.update   = csd_update,
		.clear    = csd_clear,
		.finalize = NULL,
		.snapshot = csd_snapshot,
		.source   = NRF_EDGEAI_OBSV_SOURCE_PROBS,
		.priv     = buf,
	};
}
