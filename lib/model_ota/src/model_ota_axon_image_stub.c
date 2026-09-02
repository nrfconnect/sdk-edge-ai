/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Axon partition-image stub. Raw Axon models (model_ota_axon_model) link with no nrf_edgeai_t
 * wrapper; Edge AI Lab / Axon-backend models (model_ota_edgeai_axon_model) set
 * MODEL_OTA_EDGEAI_AXON_MODEL_SRC so the image also carries solution parameters.
 */

#include <stddef.h>
#include <stdint.h>
#include <assert.h>

#define NRF_AXON_MODEL_APP_STORAGE extern

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>
#include <model_ota/model_contract.h>
#include <model_ota/model_image.h>

#if !defined(MODEL_OTA_AXON_CONFIG_VERSION) || (MODEL_OTA_AXON_CONFIG_VERSION != 1)
#error "Unsupported or missing Axon OTA configuration"
#endif

#ifndef MODEL_OTA_IMAGE_LINK_BASE
#error "MODEL_OTA_IMAGE_LINK_BASE must be defined when linking the Axon model image"
#endif

#ifndef MODEL_OTA_AXON_HEADER
#error "MODEL_OTA_AXON_HEADER is missing"
#endif

#ifndef MODEL_OTA_AXON_MODEL_SYM
#error "MODEL_OTA_AXON_MODEL_SYM is missing"
#endif

#if (MODEL_OTA_AXON_PACKED_OUTPUT_BYTES > 0) && MODEL_OTA_AXON_PACKED_OUTPUT_ALLOC
/*
 * Opt-in (model_ota_axon_model(ALLOCATE_PACKED_OUTPUT)): wire the linked model's
 * packed_output_buf field to app-owned storage, resolved via the generated PROVIDE()
 * linker fragment. Otherwise the image links with packed_output_buf NULL.
 */
#define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
#endif

#ifdef MODEL_OTA_EDGEAI_AXON_MODEL_SRC

#include STRINGIFY(MODEL_OTA_EDGEAI_AXON_MODEL_SRC)

#include "model_ota_scale_select.h"

#define MODEL_OTA_IMAGE_EDGEAI_PARAMS_INIT MODEL_OTA_IMAGE_PARAMS_INIT

#define MODEL_OTA_IMAGE_CONTRACT_HASH                                                              \
	MODEL_OTA_CONTRACT_HASH_EDGEAI_AXON(MODEL_OTA_IMAGE_LINK_BASE,                             \
					    MODEL_OTA_SOLUTION_CONTRACT_ARGS)

#else /* raw Axon */

#include MODEL_OTA_AXON_HEADER

#define MODEL_OTA_IMAGE_EDGEAI_PARAMS_INIT {{{{0}}}}
#define MODEL_OTA_IMAGE_CONTRACT_HASH MODEL_OTA_CONTRACT_HASH_AXON(MODEL_OTA_IMAGE_LINK_BASE)

#endif

#if MODEL_OTA_AXON_KEEP_SYMBOL_COUNT > 0
#define MODEL_OTA_AXON_BINDING_ENTRY(symbol) \
	{ MODEL_OTA_AXON_SYM_HASH(symbol), (const void *)(uintptr_t)&symbol },

__attribute__((section(".rodata.model_image_binding"), used))
static const struct model_image_binding_entry model_image_binding_[] = {
	MODEL_OTA_AXON_KEEP_REFS(MODEL_OTA_AXON_BINDING_ENTRY)
};
#endif

extern char __model_image_end[];

#ifndef MODEL_IMAGE_NAME_STR
#define MODEL_IMAGE_NAME_STR "axon_model"
#endif

#ifndef MODEL_IMAGE_VERSION_U32
#define MODEL_IMAGE_VERSION_U32 0x00010000u
#endif

__attribute__((section(".rodata.model_image_name"), used))
static const char model_image_name_[] = MODEL_IMAGE_NAME_STR;

__attribute__((section(".model_image.header"), used))
const struct model_image_header model_image_hdr = {
	.magic = {MODEL_IMAGE_MAGIC0, MODEL_IMAGE_MAGIC1, MODEL_IMAGE_MAGIC2, MODEL_IMAGE_MAGIC3},
	.format_version = MODEL_IMAGE_FORMAT_VERSION,
	.params_type = MODEL_IMAGE_PARAMS_AXON,
	._reserved = 0,
	.image_size = (uint32_t)((uintptr_t)&__model_image_end - MODEL_OTA_IMAGE_LINK_BASE),
	.model_version = MODEL_IMAGE_VERSION_U32,
	.contract_hash = MODEL_OTA_IMAGE_CONTRACT_HASH,
	.crc32 = 0,
	.name = model_image_name_,
	.axon = {
		.model = &MODEL_OTA_AXON_MODEL_SYM,
		.axon_packed_output_bytes = MODEL_OTA_AXON_PACKED_OUTPUT_BYTES,
		.persistent_vars_required = MODEL_OTA_AXON_PERSISTENT_VARS_REQUIRED,
#if MODEL_OTA_AXON_KEEP_SYMBOL_COUNT > 0
		.binding = model_image_binding_,
#else
		.binding = NULL,
#endif
		.binding_count = MODEL_OTA_AXON_KEEP_SYMBOL_COUNT,
	},
	.edgeai_params = MODEL_OTA_IMAGE_EDGEAI_PARAMS_INIT,
};
