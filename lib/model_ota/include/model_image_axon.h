/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_MODEL_IMAGE_AXON_H_
#define MODEL_OTA_MODEL_IMAGE_AXON_H_

/**
 * @file
 * @brief Loader for Axon model images.
 */

#include <stddef.h>
#include <stdint.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#include <model_ota/model_image_result.h>

#include "model_image.h"

#ifdef __cplusplus
extern "C" {
#endif

/** What a model slot in the firmware provides to an Axon model image. */
struct model_image_axon_expect {
	/** Hash of the firmware ABI contract. */
	uint32_t contract_hash;
	/** Persistent-vars buffer capacity, in int32_t elements. */
	uint32_t persistent_vars_cap;
	/** Packed-output buffer capacity, in bytes. */
	uint32_t packed_output_cap;
	/** Input vector list capacity, in entries. */
	uint32_t input_vector_cap;
};

/**
 * @brief Validate an Axon model image and return its compiled model.
 *
 * @param[in]  image_addr     Memory-mapped address of the model image in the model partition.
 * @param[in]  max_image_size Bytes available from @p image_addr to the end of the model partition.
 * @param[in]  expect         What the model slot provides to the model.
 * @param[out] out_model      Set to the compiled model on success.
 * @retval MODEL_IMAGE_OK on success, a negative @ref model_image_result otherwise.
 */
enum model_image_result model_image_load_axon(const uint8_t *image_addr, size_t max_image_size,
					      const struct model_image_axon_expect *expect,
					      const nrf_axon_nn_compiled_model_s **out_model);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_IMAGE_AXON_H_ */
