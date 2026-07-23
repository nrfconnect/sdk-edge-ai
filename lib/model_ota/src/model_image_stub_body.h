/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Neuton model partition-image emission.
 *
 * This file is #included at the BOTTOM of lib/model_ota/src/model_image_stub.c, right after the
 * model's own nrf_edgeai_user_model.c has been #included with MODEL_OTA_NEUTON_WIRED unset. That
 * exposes, in this same translation unit, the model's file-static compile-time descriptor
 * (`model_instance_`, with every flash pointer baked to the model's own arrays) and its data
 * arrays / output scales. Including the model source is the only way to reach those `static`
 * symbols - they are not addressable from any other translation unit, which is precisely why
 * the plain Axon "reference the model symbol from a separate stub" approach cannot be reused
 * verbatim here.
 *
 * We emit a single @ref model_image_header into section ".model_image.header". model_image.ld
 * links it first, at the partition base, followed by all reachable .rodata (the descriptor,
 * decode-output init and data). Because the image is linked at the partition base,
 * &model_instance_, &model_image_decoded_output_ and the scale arrays are already correct
 * absolute flash addresses, so they are stored directly in the header. --gc-sections drops
 * everything the header does not (transitively) reference: the runtime nrf_edgeai_t object,
 * its function pointers to app code, the DSP pipeline, input scales, etc. image_size is a
 * link-time constant from the __model_image_end anchor; crc32 is left 0 here and patched over
 * the finished binary by tools/model_ota/patch_image_crc.py.
 */

#ifndef NRF_MODEL_PARTITION_ADDR
#error "NRF_MODEL_PARTITION_ADDR must be defined when compiling the model image stub"
#endif

#include <model_ota/model_image.h>

/* Absolute-flash extent of the linked image (see model_image.ld). */
extern char __model_image_end[];

/* Map the model's MODEL_PARAMS_TYPE token (f32/q16/q8) to enum model_image_params_type, using the
 * shared mapping in model_image.h so the baked value and the app-side expectation cannot diverge.
 */
#define MODEL_IMAGE_PARAMS_TYPE_NUM MODEL_IMAGE_PARAMS_TYPE_OF(MODEL_PARAMS_TYPE)

/* Name / version default to the model's own solution id if the build does not override them. */
#ifndef MODEL_IMAGE_NAME_STR
#define MODEL_IMAGE_NAME_STR EDGEAI_LAB_SOLUTION_ID_STR
#endif
#ifndef MODEL_IMAGE_VERSION_U32
#define MODEL_IMAGE_VERSION_U32 0x00010000u
#endif

/* Full NN_DECODED_OUTPUT_INIT from the included model source (MODEL_OTA_NEUTON_WIRED unset),
 * with every meta pointer baked to absolute flash addresses inside this image.
 */
__attribute__((section(".rodata.model_image_decoded_output"), used))
static const nrf_edgeai_decoded_output_t model_image_decoded_output_ = {NN_DECODED_OUTPUT_INIT};

__attribute__((section(".model_image.header"), used))
const struct model_image_header nrf_edgeai_model_image_hdr = {
	.magic = {MODEL_IMAGE_MAGIC0, MODEL_IMAGE_MAGIC1, MODEL_IMAGE_MAGIC2, MODEL_IMAGE_MAGIC3},
	.format_version = MODEL_IMAGE_FORMAT_VERSION,
	.params_type = MODEL_IMAGE_PARAMS_TYPE_NUM,
	.task = MODEL_TASK,
	/* Whole-image byte count as a link-time constant (base .. __model_image_end). */
	.image_size = (uint32_t)((uintptr_t)&__model_image_end - (uintptr_t)NRF_MODEL_PARTITION_ADDR),
	.crc32 = 0, /* patched over the finished binary by patch_image_crc.py */
	/* Absolute flash address of the baked descriptor - a DIRECT pointer, not an offset. */
	.model.neuton = &model_instance_,
	.decoded_output = &model_image_decoded_output_,
	.name = MODEL_IMAGE_NAME_STR,
	.model_version = MODEL_IMAGE_VERSION_U32,
	.axon_packed_output_bytes = 0,
};
