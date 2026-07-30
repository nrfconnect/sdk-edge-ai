/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Edge AI Lab / Neuton-backend partition-image stub.
 */

#ifndef MODEL_OTA_IMAGE_LINK_BASE
#error "MODEL_OTA_IMAGE_LINK_BASE must be defined when compiling the Neuton model image stub"
#endif

#ifndef MODEL_IMAGE_LINK_ADDR
#error "MODEL_IMAGE_LINK_ADDR must be defined when compiling the Neuton model image stub"
#endif

#ifndef MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC
#error "MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC must be defined by model_ota_edgeai_neuton_model()"
#endif

#include <zephyr/toolchain.h>

#include STRINGIFY(MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC)

#include <model_ota/model_contract.h>
#include <model_ota/model_image.h>

#include "model_ota_scale_select.h"

extern char __model_image_end[];

#define MODEL_IMAGE_PARAMS_TYPE_NUM MODEL_IMAGE_PARAMS_TYPE_OF(MODEL_PARAMS_TYPE)

#ifndef MODEL_IMAGE_NAME_STR
#define MODEL_IMAGE_NAME_STR EDGEAI_LAB_SOLUTION_ID_STR
#endif
#ifndef MODEL_IMAGE_VERSION_U32
#define MODEL_IMAGE_VERSION_U32 0x00010000u
#endif

#define MODEL_OTA_EDGEAI_NEUTON_CONTRACT_HASH                                                      \
	MODEL_OTA_CONTRACT_HASH_EDGEAI_NEUTON(MODEL_OTA_IMAGE_LINK_BASE,                           \
					      MODEL_IMAGE_PARAMS_TYPE_NUM,                         \
					      MODEL_OTA_SOLUTION_CONTRACT_ARGS)

__attribute__((section(".rodata.model_image_name"), used))
static const char model_image_name_[] = MODEL_IMAGE_NAME_STR;

__attribute__((section(".model_image.header"), used))
const struct model_image_header nrf_edgeai_model_image_hdr = {
	.magic = {MODEL_IMAGE_MAGIC0, MODEL_IMAGE_MAGIC1, MODEL_IMAGE_MAGIC2, MODEL_IMAGE_MAGIC3},
	.format_version = MODEL_IMAGE_FORMAT_VERSION,
	.params_type = MODEL_IMAGE_PARAMS_TYPE_NUM,
	._reserved = 0,
	.image_size = (uint32_t)((uintptr_t)&__model_image_end - (uintptr_t)MODEL_IMAGE_LINK_ADDR),
	.model_version = MODEL_IMAGE_VERSION_U32,
	.contract_hash = MODEL_OTA_EDGEAI_NEUTON_CONTRACT_HASH,
	.crc32 = 0,
	.name = model_image_name_,
	.neuton = {
		.model = &model_instance_,
	},
	.edgeai_params = MODEL_OTA_IMAGE_PARAMS_INIT,
};
