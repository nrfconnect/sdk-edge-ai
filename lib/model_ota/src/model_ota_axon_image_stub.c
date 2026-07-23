/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Axon model partition-image stub (one translation unit, compiled once per model image).
 *
 * model_ota_axon_image() compiles this file with MODEL_IMAGE_HEADER and MODEL_IMAGE_MODEL_SYM,
 * then links the result at the partition base. App-owned pointer fields are resolved from
 * zephyr.elf via a generated PROVIDE() linker fragment.
 */

#include "model_ota_stub_macros.h"

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#define NRF_AXON_MODEL_APP_STORAGE extern

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <model_ota/model_image.h>

#ifndef NRF_MODEL_PARTITION_ADDR
#error "NRF_MODEL_PARTITION_ADDR must be defined when linking the Axon model image"
#endif

#ifndef MODEL_IMAGE_HEADER
#error "MODEL_IMAGE_HEADER must be defined when building the Axon model image stub"
#endif

#ifndef MODEL_IMAGE_MODEL_SYM
#error "MODEL_IMAGE_MODEL_SYM must be defined when building the Axon model image stub"
#endif

#include MODEL_IMAGE_HEADER

extern char __model_image_end[];

#ifndef MODEL_IMAGE_NAME_STR
#define MODEL_IMAGE_NAME_STR "axon_model"
#endif

#ifndef MODEL_IMAGE_VERSION_U32
#define MODEL_IMAGE_VERSION_U32 0x00010000u
#endif

#ifndef MODEL_IMAGE_PACKED_OUTPUT_BYTES
#define MODEL_IMAGE_PACKED_OUTPUT_BYTES 0u
#endif

static const union model_image_model_ptr model_image_model = {
	.axon = &MODEL_IMAGE_MODEL_SYM,
};

__attribute__((section(".model_image.header"), used))
const struct model_image_header model_image_hdr = {
	.magic = {MODEL_IMAGE_MAGIC0, MODEL_IMAGE_MAGIC1, MODEL_IMAGE_MAGIC2, MODEL_IMAGE_MAGIC3},
	.format_version = MODEL_IMAGE_FORMAT_VERSION,
	.params_type = MODEL_IMAGE_PARAMS_AXON,
	.task = 0,
	.image_size = (uint32_t)((uintptr_t)&__model_image_end - (uintptr_t)NRF_MODEL_PARTITION_ADDR),
	.crc32 = 0,
	.model = model_image_model,
	.decoded_output = NULL,
	.name = MODEL_IMAGE_NAME_STR,
	.model_version = MODEL_IMAGE_VERSION_U32,
	.axon_packed_output_bytes = MODEL_IMAGE_PACKED_OUTPUT_BYTES,
};
