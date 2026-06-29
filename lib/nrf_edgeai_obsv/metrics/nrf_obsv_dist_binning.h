/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef NRF_OBSV_DIST_BINNIG_H_
#define NRF_OBSV_DIST_BINNIG_H_

#include <stdint.h>

/*
 * Shared histogram-binning helpers for distribution-style metrics.
 *
 * Two binning paths are provided:
 *
 *  - Custom edges: a metric that needs arbitrary (non-uniform) bin boundaries
 *    stores bin_num - 1 ascending inner edges and bins values with
 *    _dist_find_bin(). The per-class probability distribution metric uses this.
 *
 *  - Uniform [0, 1] fast path: a metric whose bins are always uniform over
 *    [0, 1] needs no stored edges and bins values in O(1) with
 *    _dist_uniform_bin(). The entropy/margin/mel descriptor metrics use this.
 */

/* Fill the bin_num - 1 inner edges of bin_num uniform bins spanning [0, 1]. */
void _dist_uniform_edges(float *edges, uint8_t bin_num);

/*
 * Return the bin index in [0, bin_num - 1] that @val falls into, given the
 * bin_num - 1 ascending inner edges. A value below edges[b] lands in bin b;
 * a value at or above the last edge lands in the top bin.
 */
uint8_t _dist_find_bin(const float *edges, uint8_t bin_num, float val);

/*
 * Return the bin index in [0, bin_num - 1] for @val over bin_num uniform bins
 * spanning [0, 1], without any stored edges or search. @val <= 0 lands in bin 0;
 * @val >= 1 lands in the top bin. Equivalent to _dist_find_bin() against uniform
 * edges, but O(1).
 */
uint8_t _dist_uniform_bin(uint8_t bin_num, float val);

/*
 * Clip @v to [0, 1]. Used by the mel descriptor metrics to clamp their
 * scale-invariant statistics to [0, 1] before binning.
 */
static inline float _clip01(float v)
{
	return (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
}

#endif /* NRF_OBSV_DIST_BINNIG_H_ */
