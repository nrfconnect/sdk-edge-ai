/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Neuton model partition-image stub (one translation unit, compiled once per model image).
 *
 * model_ota_neuton_image() sets MODEL_OTA_NEUTON_MODEL_SRC and NRF_MODEL_PARTITION_ADDR,
 * then compiles this file as an OBJECT library. MODEL_OTA_NEUTON_WIRED is not set, so the
 * included nrf_edgeai_user_model.c emits the compile-time model_instance_ descriptor and
 * payload arrays; this stub then emits the partition header that roots the gc-sections link.
 */

#include "model_ota_stub_macros.h"

#ifndef NRF_MODEL_PARTITION_ADDR
#error "NRF_MODEL_PARTITION_ADDR must be defined when compiling the Neuton model image stub"
#endif

#ifndef MODEL_OTA_NEUTON_MODEL_SRC
#error "MODEL_OTA_NEUTON_MODEL_SRC must be defined by model_ota_neuton_image()"
#endif

#include MODEL_OTA_STUB_STR(MODEL_OTA_NEUTON_MODEL_SRC)

/*
 * Partition-image emission (included model must expose file-static model_instance_ and data).
 *
 * We emit a single @ref model_image_header into section ".model_image.header". model_image.ld
 * links it first, at the partition base, followed by all reachable .rodata (the descriptor,
 * decode-output init and data). Because the image is linked at the partition base,
 * &model_instance_, &model_image_decoded_output_ and the scale arrays are already correct
 * absolute flash addresses, so they are stored directly in the header. --gc-sections drops
 * everything the header does not (transitively) reference. image_size is a link-time constant
 * from the __model_image_end anchor; crc32 is left 0 and patched by patch_image_crc.py.
 */

#include <model_ota/model_image.h>

extern char __model_image_end[];

#define MODEL_IMAGE_PARAMS_TYPE_NUM MODEL_IMAGE_PARAMS_TYPE_OF(MODEL_PARAMS_TYPE)

#ifndef MODEL_IMAGE_NAME_STR
#define MODEL_IMAGE_NAME_STR EDGEAI_LAB_SOLUTION_ID_STR
#endif
#ifndef MODEL_IMAGE_VERSION_U32
#define MODEL_IMAGE_VERSION_U32 0x00010000u
#endif

__attribute__((section(".rodata.model_image_decoded_output"), used))
static const nrf_edgeai_decoded_output_t model_image_decoded_output_ = {NN_DECODED_OUTPUT_INIT};

__attribute__((section(".model_image.header"), used))
const struct model_image_header nrf_edgeai_model_image_hdr = {
	.magic = {MODEL_IMAGE_MAGIC0, MODEL_IMAGE_MAGIC1, MODEL_IMAGE_MAGIC2, MODEL_IMAGE_MAGIC3},
	.format_version = MODEL_IMAGE_FORMAT_VERSION,
	.params_type = MODEL_IMAGE_PARAMS_TYPE_NUM,
	.task = MODEL_TASK,
	.image_size = (uint32_t)((uintptr_t)&__model_image_end - (uintptr_t)NRF_MODEL_PARTITION_ADDR),
	.crc32 = 0,
	.model.neuton = &model_instance_,
	.decoded_output = &model_image_decoded_output_,
	.name = MODEL_IMAGE_NAME_STR,
	.model_version = MODEL_IMAGE_VERSION_U32,
	.axon_packed_output_bytes = 0,
};
