/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Fixed stub for Axon model partition images.
 *
 * Instead of generating a multi-megabyte *_model_image.c that copies model bytes,
 * this file is compiled once per application. The compiler-generated model header
 * is pulled in via the generated fixups header (gen_axon_model_partition_fixups.py).
 *
 * Pointer fields that reference application RAM/flash are not computed here; they
 * are supplied by a generated linker script built from zephyr.elf symbol addresses.
 */

#include <stddef.h>
#include <stdint.h>

#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <axon/nrf_axon_model_image_layout.h>
#include <axon/nrf_axon_model_partition_defs.h>

#ifndef NRF_AXON_MODEL_PARTITION_ADDR
#error "NRF_AXON_MODEL_PARTITION_ADDR must be defined when linking the model image"
#endif

#ifndef NRF_AXON_MODEL_IMAGE_MODEL_SYM
#error "NRF_AXON_MODEL_IMAGE_MODEL_SYM must be defined in the generated fixups header"
#endif

#define MODEL_IMAGE_MODEL_SYM NRF_AXON_MODEL_IMAGE_MODEL_SYM

/*
 * Partition header is linked first (see model_image.ld).
 *
 * model_offset and image_size are link-time constants expressed relative to
 * NRF_AXON_MODEL_PARTITION_ADDR. Post-link validation compares them against
 * __axon_model_image_* anchors and the compiled model symbol.
 */
__attribute__((section(".model_image.partition_hdr"), used))
const struct nrf_axon_model_partition_header nrf_axon_model_image_partition_hdr = {
	.magic = NRF_AXON_MODEL_PARTITION_MAGIC,
	.version = NRF_AXON_MODEL_PARTITION_VERSION,
	.model_offset = (uint32_t)((uintptr_t)&MODEL_IMAGE_MODEL_SYM -
				  (uintptr_t)NRF_AXON_MODEL_PARTITION_ADDR),
	.image_size = (uint32_t)((uintptr_t)&__axon_model_image_end -
				 (uintptr_t)NRF_AXON_MODEL_PARTITION_ADDR),
};
