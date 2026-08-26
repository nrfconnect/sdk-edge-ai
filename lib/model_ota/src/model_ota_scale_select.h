/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Picks the model-owned parameters out of a generated Edge AI Lab solution source that has
 * already been #included, for both the image stub (which bakes them) and the wired application
 * translation unit (which states what it expects).
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

#include <stdint.h>

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

/** Initializer for struct model_image_scale_expect (app side). */
#define MODEL_OTA_SCALE_EXPECT_INIT                                                                \
	{                                                                                          \
		.num = MODEL_OTA_SCALE_NUM,                                                        \
		.elem_size = MODEL_OTA_SCALE_ELEM_SIZE,                                            \
	}

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
		.scale_num = MODEL_OTA_SCALE_NUM,                                                  \
		.scale_elem_size = MODEL_OTA_SCALE_ELEM_SIZE,                                      \
	}

#endif /* MODEL_OTA_WIRED */

#endif /* MODEL_OTA_SCALE_SELECT_H_ */
