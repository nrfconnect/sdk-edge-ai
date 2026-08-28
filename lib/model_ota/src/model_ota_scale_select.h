/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Picks the model-owned parameters out of a generated Edge AI Lab solution source that has
 * already been #included, for both the image stub (which bakes them) and the wired application
 * translation unit (which loads them), and gathers the solution's contract-hash arguments.
 *
 * A solution scales once on the way into the network: without a DSP feature pipeline it scales the
 * raw input features (INPUT_FEATURES_SCALE_*), with one it scales the extracted features
 * (EXTRACTED_FEATURES_SCALE_*, alongside the extraction arguments). Those are the factors a model
 * update has to be able to replace; the other stage's, if present, belong to the application.
 *
 * The image stub and the wired application translation unit both include this header after the
 * model source, so the two sides of an update derive the layout from the same macros and agree by
 * construction. This holds for either backend: what a solution keeps in nrf_edgeai_t does not
 * depend on whether its network is Neuton or Axon. The pointer-bearing initializer is only
 * available where the model still initializes from its arrays, i.e. when MODEL_OTA_WIRED is not
 * set.
 */

#ifndef MODEL_OTA_SCALE_SELECT_H_
#define MODEL_OTA_SCALE_SELECT_H_

#include <stddef.h>
#include <stdint.h>

#include <model_ota/model_contract.h>

#if !defined(EXTRACTED_FEATURES_NUM) || !defined(INPUT_UNIQUE_SCALES_NUM)
#error "model_ota_scale_select.h must be included after the generated model source"
#endif

#if EXTRACTED_FEATURES_NUM > 0
#define MODEL_OTA_SCALE_NUM	  EXTRACTED_FEATURES_NUM
#define MODEL_OTA_SCALE_ELEM_SIZE ((uint8_t)sizeof(nrf_user_feature_t))
#else
#define MODEL_OTA_SCALE_NUM	  INPUT_UNIQUE_SCALES_NUM
#define MODEL_OTA_SCALE_ELEM_SIZE ((uint8_t)sizeof(nrf_user_input_t))
#endif

#ifndef MODEL_OTA_SOLUTION_ID_HASH
#error "MODEL_OTA_SOLUTION_ID_HASH must be defined by the model_ota CMake helper"
#endif

/*
 * FEATURES_EXTRACTION_ARGUMENTS is either a generated array or, for a pipeline that takes no
 * arguments, the placeholder `#define FEATURES_EXTRACTION_ARGUMENTS NULL`. Being defined as a
 * macro is therefore exactly what marks the empty case - and the case sizeof() cannot describe.
 */
#ifdef FEATURES_EXTRACTION_ARGUMENTS
#define MODEL_OTA_DSP_ARGS_BYTES     0u
#define MODEL_OTA_DSP_ARGS_ELEM_SIZE 0u
#else
#define MODEL_OTA_DSP_ARGS_BYTES     ((uint32_t)sizeof(FEATURES_EXTRACTION_ARGUMENTS))
#define MODEL_OTA_DSP_ARGS_ELEM_SIZE ((uint32_t)sizeof(FEATURES_EXTRACTION_ARGUMENTS[0]))
#endif

/* FFT geometry exists only for a solution with a frequency-domain pipeline. */
#ifdef DSP_RFFT_LEN
#define MODEL_OTA_DSP_SPECTRUM_LEN ((uint32_t)DSP_AMPLITUDE_SPECTRUM_LEN)
#define MODEL_OTA_DSP_RFFT_LEN	   ((uint32_t)DSP_RFFT_LEN)
#define MODEL_OTA_DSP_BITREV_LEN   ((uint32_t)DSP_CFFT_BITREV_INDEX_TABLE_LEN)
#else
#define MODEL_OTA_DSP_SPECTRUM_LEN 0u
#define MODEL_OTA_DSP_RFFT_LEN	   0u
#define MODEL_OTA_DSP_BITREV_LEN   0u
#endif

#if EXTRACTED_FEATURES_NUM > 0
/** Digest of the DSP feature-extraction contract; see @ref MODEL_OTA_HASH_DSP_MIX. */
#define MODEL_OTA_DSP_CONTRACT_HASH                                                                \
	MODEL_OTA_CONTRACT_HASH_DSP(MODEL_OTA_DSP_ARGS_BYTES, MODEL_OTA_DSP_ARGS_ELEM_SIZE,         \
				    MODEL_OTA_DSP_SPECTRUM_LEN, MODEL_OTA_DSP_RFFT_LEN,            \
				    MODEL_OTA_DSP_BITREV_LEN)

/** The array the loader compares against the application's, see @ref MODEL_OTA_HASH_DSP_MIX. */
#define MODEL_OTA_EXTRACTION_MASK_PTR                                                              \
	((const nrf_edgeai_features_mask_t *)FEATURES_EXTRACTION_MASK)
#else
/* No DSP pipeline: nothing to describe, and the generator emits neither array. */
#define MODEL_OTA_DSP_CONTRACT_HASH   0u
#define MODEL_OTA_EXTRACTION_MASK_PTR NULL
#endif

/**
 * The solution-derived arguments of @ref MODEL_OTA_HASH_SOLUTION_MIX, in its order.
 *
 * Both sides of an update expand this after the same generated model source, so they agree by
 * construction. MODEL_OTA_SOLUTION_ID_HASH is a hash of the solution ID, computed at configure
 * time (model_ota_solution_id_hash() in model_ota_common.cmake) because a string cannot be hashed
 * by the preprocessor.
 *
 * MODEL_USES_AS_INPUT_MASK is the generated packing of the per-stage
 * MODEL_USES_AS_INPUT_INPUT_FEATURES / _DSP_FEATURES flags, and is what the runtime itself stores
 * in nrf_edgeai_t.model.uses_as_input.all - so it is the form to hash rather than its components.
 */
#define MODEL_OTA_SOLUTION_CONTRACT_ARGS                                                           \
	MODEL_OTA_SOLUTION_ID_HASH, EDGEAI_RUNTIME_VERSION_COMBINED, MODEL_TASK,                   \
		MODEL_OUTPUTS_NUM, MODEL_OTA_SCALE_NUM, MODEL_OTA_SCALE_ELEM_SIZE,                  \
		INPUT_FEATURE_DATA_TYPE, INPUT_UNIQ_FEATURES_NUM, INPUT_WINDOW_SIZE,                \
		INPUT_WINDOW_SHIFT, INPUT_SUBWINDOW_NUM, EXTRACTED_FEATURES_NUM,                    \
		MODEL_USES_AS_INPUT_MASK, MODEL_OTA_DSP_CONTRACT_HASH

#ifndef MODEL_OTA_WIRED

#if EXTRACTED_FEATURES_NUM > 0
#define MODEL_OTA_IMAGE_SCALE_INIT                                                                 \
	.features.EXTRACTED_FEATURES_META_TYPE = {                                                 \
		.p_min = EXTRACTED_FEATURES_SCALE_MIN,                                             \
		.p_max = EXTRACTED_FEATURES_SCALE_MAX,                                             \
		.p_arguments = FEATURES_EXTRACTION_ARGUMENTS,                                      \
	}
#else
#define MODEL_OTA_IMAGE_SCALE_INIT                                                                 \
	.input.INPUT_TYPE = {                                                                      \
		.p_min = INPUT_FEATURES_SCALE_MIN,                                                 \
		.p_max = INPUT_FEATURES_SCALE_MAX,                                                 \
	}
#endif

/** Initializer for the struct model_image_edgeai_params baked into the image. */
#define MODEL_OTA_IMAGE_PARAMS_INIT                                                                \
	{                                                                                          \
		.scale = {MODEL_OTA_IMAGE_SCALE_INIT},                                             \
		.decoded_output = {NN_DECODED_OUTPUT_INIT},                                        \
		.p_extraction_mask = MODEL_OTA_EXTRACTION_MASK_PTR,                                \
		.scale_num = MODEL_OTA_SCALE_NUM,                                                  \
		.scale_elem_size = MODEL_OTA_SCALE_ELEM_SIZE,                                      \
	}

#endif /* MODEL_OTA_WIRED */

#endif /* MODEL_OTA_SCALE_SELECT_H_ */
