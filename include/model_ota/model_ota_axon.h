/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @addtogroup model_ota
 * @{
 */

#ifndef MODEL_OTA_MODEL_OTA_AXON_H_
#define MODEL_OTA_MODEL_OTA_AXON_H_

/**
 * @file
 * @brief Loaders of Axon model slots.
 */

#include <drivers/axon/nrf_axon_nn_infer.h>

#include <model_ota/model_image_result.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_OTA_AXON_LOAD_DECL_(target)                                                          \
	enum model_image_result model_ota_load_axon_##target(                                      \
		const nrf_axon_nn_compiled_model_s **out_model)

/**
 * @brief Declare the loader of the model slot @p target.
 *
 * model_ota_axon_model() in lib/model_ota/cmake/backends/axon.cmake defines the loader
 * model_ota_load_axon_<target>(), which validates the model image in the model partition of the
 * model slot and returns its compiled model:
 *
 * @code{.c}
 * enum model_image_result model_ota_load_axon_<target>(
 *	const nrf_axon_nn_compiled_model_s **out_model);
 * @endcode
 *
 * @p out_model is set to the compiled model on success, or NULL on failure. The loader returns
 * @ref MODEL_IMAGE_OK on success, a negative @ref model_image_result otherwise.
 *
 * @param target Model slot ID, the TARGET passed to model_ota_axon_model().
 */
#define MODEL_OTA_AXON_LOAD_DECL(target) MODEL_OTA_AXON_LOAD_DECL_(target)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_OTA_AXON_H_ */

/**
 * @}
 */
