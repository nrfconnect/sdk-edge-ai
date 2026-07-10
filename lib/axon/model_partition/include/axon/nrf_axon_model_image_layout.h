/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <stdint.h>

#include <axon/nrf_axon_model_partition_defs.h>

/*
 * Linker-defined anchors for the model partition image (see linker/model_image.ld).
 * Addresses are absolute flash locations when the image is linked at
 * NRF_AXON_MODEL_PARTITION_ADDR.
 */
extern char __axon_model_image_start[];
extern char __axon_model_image_end[];

static inline uintptr_t nrf_axon_model_image_base(void)
{
	return (uintptr_t)__axon_model_image_start;
}

static inline uint32_t nrf_axon_model_image_size(void)
{
	return (uint32_t)((uintptr_t)__axon_model_image_end -
			  (uintptr_t)__axon_model_image_start);
}

static inline uint32_t nrf_axon_model_image_model_offset(const void *model)
{
	return (uint32_t)((uintptr_t)model - (uintptr_t)__axon_model_image_start);
}
