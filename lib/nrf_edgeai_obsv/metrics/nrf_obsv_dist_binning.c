/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "nrf_obsv_dist_binning.h"

void _dist_uniform_edges(float *edges, uint8_t bin_num)
{
	const float step = 1.0f / bin_num;

	for (int i = 0; i < bin_num - 1; i++) {
		edges[i] = (i + 1) * step;
	}
}

uint8_t _dist_find_bin(const float *edges, uint8_t bin_num, float val)
{
	for (uint8_t b = 0; b < bin_num - 1; b++) {
		if (val < edges[b]) {
			return b;
		}
	}
	return bin_num - 1;
}

uint8_t _dist_uniform_bin(uint8_t bin_num, float val)
{
	if (val <= 0.0f) {
		return 0;
	}
	if (val >= 1.0f) {
		return (uint8_t)(bin_num - 1);
	}

	uint8_t b = (uint8_t)(val * (float)bin_num);

	return (b < bin_num) ? b : (uint8_t)(bin_num - 1);
}
