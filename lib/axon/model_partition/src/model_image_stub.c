/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Fixed stub for Axon model partition images. Includes the compiler-generated
 * model header directly; pointer fixups for app-owned symbols come from
 * model_image_fixups.h (small generated header) and syms.h (link-time addresses).
 */

#include <stddef.h>
#include <stdint.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_model_partition_defs.h>

#ifndef NRF_AXON_MODEL_PARTITION_ADDR
#error "NRF_AXON_MODEL_PARTITION_ADDR must be defined when linking the model image"
#endif

#ifndef NRF_AXON_MODEL_IMAGE_MODEL_SYM
#error "NRF_AXON_MODEL_IMAGE_MODEL_SYM must be defined in the generated fixups header"
#endif

#define MODEL_IMAGE_MODEL_SYM NRF_AXON_MODEL_IMAGE_MODEL_SYM

/*
 * Partition header is linked first (see model_image.ld), before model rodata.
 * model_offset and image_size use the partition base passed at link time.
 */
__attribute__((section(".model_image.partition_hdr"), used))
const struct nrf_axon_model_partition_header nrf_axon_model_image_partition_hdr = {
	.magic = NRF_AXON_MODEL_PARTITION_MAGIC,
	.version = NRF_AXON_MODEL_PARTITION_VERSION,
	.model_offset = (uint32_t)((uintptr_t)&MODEL_IMAGE_MODEL_SYM -
				  (uintptr_t)NRF_AXON_MODEL_PARTITION_ADDR),
	.image_size = (uint32_t)((uintptr_t)&MODEL_IMAGE_MODEL_SYM +
				 sizeof(MODEL_IMAGE_MODEL_SYM) -
				 (uintptr_t)NRF_AXON_MODEL_PARTITION_ADDR),
};
