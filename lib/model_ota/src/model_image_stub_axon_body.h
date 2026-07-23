/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Axon model partition-image emission.
 *
 * Included at the bottom of model_image_stub_axon.c after MODEL_IMAGE_HEADER pulls in
 * the compiler-generated model. Emits a @ref model_image_header with a DIRECT pointer to
 * the baked nrf_axon_nn_compiled_model_s and NULL decoded_output.
 */

#ifndef NRF_MODEL_PARTITION_ADDR
#error "NRF_MODEL_PARTITION_ADDR must be defined when compiling the Axon model image stub"
#endif

#ifndef MODEL_IMAGE_MODEL_SYM
#error "MODEL_IMAGE_MODEL_SYM must be defined when building the Axon model image stub"
#endif

#include <model_ota/model_image.h>

extern char __model_image_end[];

#ifndef MODEL_IMAGE_NAME_STR
#define MODEL_IMAGE_NAME_STR "axon_model"
#endif

#ifndef MODEL_IMAGE_VERSION_U32
#define MODEL_IMAGE_VERSION_U32 0x00010000u
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
};
