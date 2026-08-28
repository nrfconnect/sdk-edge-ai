/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Axon model partition-image stub (one translation unit, compiled once per model image).
 *
 * model_ota_axon_model() compiles this file with probe-derived configuration,
 * then links the result at the partition base. App-owned pointer fields are
 * resolved from zephyr.elf via a generated PROVIDE() linker fragment.
 *
 * Shared by both flavours of Axon model. For an Edge AI Lab solution with an Axon backend,
 * model_ota_axon_edgeai_wire() also passes MODEL_OTA_AXON_EDGEAI_MODEL_SRC, and the image then
 * carries the solution's nrf_edgeai_t parameters as well as the compiled model. A pure Axon model
 * has no nrf_edgeai_t, so its header leaves that block zeroed.
 *
 * TODO (potential): separate the pure Axon and Axon-backed-Lab paths further. Today they differ
 * only in the #ifdef below and in the optional EDGEAI_MODEL_SRC argument of
 * model_ota_axon_model(), while sharing the whole image pipeline (probe, axon_config.h, binding
 * table, PROVIDE pass, link, CRC, validation) - so a split is not obviously worth the duplicated
 * header definition, and the fork has to live in this file regardless, because edgeai_params is
 * an in-place member whose initializer must be a compile-time constant in the translation unit
 * that defines the header. If a second point of divergence appears, the least-cost form is a
 * model_ota_axon_edgeai_image_stub.c that #includes a shared header-emission part, keeping one
 * definition of the on-flash layout.
 */

#include "model_ota_stub_macros.h"

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

#ifndef NRF_MODEL_PARTITION_ADDR
#error "NRF_MODEL_PARTITION_ADDR must be defined when linking the Axon model image"
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

#ifdef MODEL_OTA_AXON_EDGEAI_MODEL_SRC
/*
 * Edge AI Lab solution: pull in the whole generated solution source, which #includes the Axon
 * model header itself. MODEL_OTA_WIRED is deliberately *not* set here - this translation unit is
 * the payload, so the compiled model and the scaling / decode arrays must all be emitted, and
 * model_ota_scale_select.h below turns them into the header's parameter block. --gc-sections then
 * drops everything the header does not reference, including the solution's own nrf_edgeai_t.
 */
#include STRINGIFY(MODEL_OTA_AXON_EDGEAI_MODEL_SRC)

#include "model_ota_scale_select.h"

#define MODEL_OTA_AXON_EDGEAI_PARAMS_INIT MODEL_OTA_IMAGE_PARAMS_INIT

/*
 * A wrapped solution hashes its whole nrf_edgeai_t contract on top of the Axon one, which is what
 * keeps a pure Axon image out of this slot: such an image carries no edgeai_params, while an
 * OTA-wired application has discarded its own compiled-in copy.
 */
#define MODEL_OTA_AXON_CONTRACT_HASH                                                               \
	MODEL_OTA_CONTRACT_HASH_AXON_EDGEAI(NRF_MODEL_PARTITION_ADDR,                              \
					    MODEL_OTA_SOLUTION_CONTRACT_ARGS)
#else
#include MODEL_OTA_AXON_HEADER

/* Pure Axon model: no nrf_edgeai_t to carry. */
#define MODEL_OTA_AXON_EDGEAI_PARAMS_INIT {0}

#define MODEL_OTA_AXON_CONTRACT_HASH MODEL_OTA_CONTRACT_HASH_AXON(NRF_MODEL_PARTITION_ADDR)
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
	.image_size = (uint32_t)((uintptr_t)&__model_image_end - (uintptr_t)NRF_MODEL_PARTITION_ADDR),
	.model_version = MODEL_IMAGE_VERSION_U32,
	.contract_hash = MODEL_OTA_AXON_CONTRACT_HASH,
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
	.edgeai_params = MODEL_OTA_AXON_EDGEAI_PARAMS_INIT,
};
