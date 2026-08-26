/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_CONTRACT_H_
#define MODEL_OTA_MODEL_CONTRACT_H_

/* TODO: improve contract hash calculation */
/* TODO: resolve part of issues from security review on confluence */

/**
 * @file
 * @brief Firmware ABI contract hashing for model-only OTA images.
 *
 * Image and application must compute the same @ref model_image_header.contract_hash
 * from compile-time constants. The hash is FNV-1a over typed u32 chunks (little-endian
 * identity on the target).
 */

#include <stdint.h>

#include <model_ota/model_image.h>

#include <nrf_edgeai/rt/nrf_edgeai_model_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_output_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Contract backend discriminator mixed into every hash. */
#define MODEL_OTA_CONTRACT_BACKEND_NEUTON 0u
#define MODEL_OTA_CONTRACT_BACKEND_AXON   1u

#define MODEL_OTA_FNV1A_INIT 2166136261u
#define MODEL_OTA_FNV1A_MUL  16777619u

/** One FNV-1a step mixing a 32-bit value. Usable in static initializers. */
#define MODEL_OTA_FNV1A_U32(h, v) \
	(((uint32_t)(h) ^ (uint32_t)(v)) * (uint32_t)MODEL_OTA_FNV1A_MUL)

/** FNV-1a over a NUL-terminated ASCII string (for solution id). */
#define MODEL_OTA_FNV1A_STR(h, s) MODEL_OTA_FNV1A_STR_(h, s)
#define MODEL_OTA_FNV1A_STR_(h, s) MODEL_OTA_FNV1A_STR__(h, s, 0)
#define MODEL_OTA_FNV1A_STR__(h, s, i) \
	((s)[(i)] == '\0' ? (h) : MODEL_OTA_FNV1A_STR__(MODEL_OTA_FNV1A_U32((h), (s)[(i)]), s, (i) + 1))

/**
 * Neuron scratch element size for @p params_type (f32/q16/q8 enum or token).
 */
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE(params_type) \
	MODEL_OTA_NEUTON_NEURON_ELEM_SIZE_(params_type)
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE_(params_type) \
	MODEL_OTA_NEUTON_NEURON_ELEM_SIZE__##params_type
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE__0 4u
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE__1 2u
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE__2 1u
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE__MODEL_IMAGE_PARAMS_F32 4u
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE__MODEL_IMAGE_PARAMS_Q16 2u
#define MODEL_OTA_NEUTON_NEURON_ELEM_SIZE__MODEL_IMAGE_PARAMS_Q8 1u

/** DSP / input pipeline identity for a Neuton solution. */
#define MODEL_OTA_NEUTON_PIPELINE_HASH( \
	input_feature_type, window_size, window_shift, uniq_features, uses_input, uses_dsp) \
	MODEL_OTA_FNV1A_U32( \
		MODEL_OTA_FNV1A_U32( \
			MODEL_OTA_FNV1A_U32( \
				MODEL_OTA_FNV1A_U32( \
					MODEL_OTA_FNV1A_U32( \
						MODEL_OTA_FNV1A_U32(MODEL_OTA_FNV1A_INIT, (input_feature_type)), \
						(window_size)), \
					(window_shift)), \
				(uniq_features)), \
			MODEL_OTA_FNV1A_U32((uses_input), (uses_dsp)))

/**
 * Full Neuton contract hash. @p params_type is @ref model_image_params_type (0/1/2).
 *
 * @p solution_id_str   EDGEAI_LAB_SOLUTION_ID_STR
 * @p neurons_cap       MODEL_OTA_NEUTON_NEURONS_CAP
 */
#define MODEL_OTA_CONTRACT_HASH_NEUTON( \
	task, params_type, outputs_cap, inputs_num, neurons_cap, solution_id_str, pipeline_hash) \
	MODEL_OTA_FNV1A_STR( \
		MODEL_OTA_FNV1A_U32( \
			MODEL_OTA_FNV1A_U32( \
				MODEL_OTA_FNV1A_U32( \
					MODEL_OTA_FNV1A_U32( \
						MODEL_OTA_FNV1A_U32( \
							MODEL_OTA_FNV1A_U32( \
								MODEL_OTA_FNV1A_U32( \
									MODEL_OTA_FNV1A_U32( \
										MODEL_OTA_FNV1A_U32( \
											MODEL_OTA_FNV1A_U32( \
												MODEL_OTA_FNV1A_U32( \
													MODEL_OTA_FNV1A_INIT, \
													MODEL_IMAGE_FORMAT_VERSION), \
												MODEL_OTA_CONTRACT_BACKEND_NEUTON), \
											(task)), \
										(params_type)), \
									((uint32_t)sizeof(nrf_edgeai_model_neuton_t))), \
								((uint32_t)sizeof(nrf_nn_neuton_model_meta_t))), \
							((uint32_t)sizeof(nrf_edgeai_decoded_output_t))), \
						MODEL_OTA_NEUTON_NEURON_ELEM_SIZE(params_type)), \
					(outputs_cap)), \
				(inputs_num)), \
			(neurons_cap)), \
		(pipeline_hash)), \
		(solution_id_str))

/**
 * Full Axon contract hash.
 *
 * TODO: an Axon-backed Edge AI Lab solution now carries nrf_edgeai_t parameters as well as the
 * compiled model, so this should cover the solution's task and id the way
 * @ref MODEL_OTA_CONTRACT_HASH_NEUTON does (which is where a Neuton image's task is validated,
 * now that the header no longer carries one).
 *
 * @p persistent_required  Elements (int32_t) required in persistent_vars (from probe).
 * @p packed_output_bytes  Bytes required for packed output (0 when unused).
 */
#define MODEL_OTA_CONTRACT_HASH_AXON(persistent_required, packed_output_bytes) \
	MODEL_OTA_FNV1A_U32( \
		MODEL_OTA_FNV1A_U32( \
			MODEL_OTA_FNV1A_U32( \
				MODEL_OTA_FNV1A_U32( \
					MODEL_OTA_FNV1A_U32( \
						MODEL_OTA_FNV1A_U32( \
							MODEL_OTA_FNV1A_U32(MODEL_OTA_FNV1A_INIT, \
									    MODEL_IMAGE_FORMAT_VERSION), \
							MODEL_OTA_CONTRACT_BACKEND_AXON), \
						((uint32_t)sizeof(nrf_axon_nn_compiled_model_s))), \
					(uint32_t)CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE), \
				(uint32_t)CONFIG_NRF_AXON_PSUM_BUFFER_SIZE), \
			(persistent_required)), \
		(packed_output_bytes))

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_CONTRACT_H_ */
