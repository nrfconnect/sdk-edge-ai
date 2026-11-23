/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup model_ota Model OTA
 * @{
 *
 * @brief Update models without updating the firmware, using model images in model partitions.
 *
 */
#ifndef MODEL_OTA_MODEL_IMAGE_RESULT_H_
#define MODEL_OTA_MODEL_IMAGE_RESULT_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Return codes of the model image loaders.
 */
enum model_image_result {
	/** The image is valid. */
	MODEL_IMAGE_OK = 0,
	/** A required argument is missing or invalid. */
	MODEL_IMAGE_ERR_INVALID_ARG = -1,
	/** The partition holds no model image. */
	MODEL_IMAGE_ERR_BAD_MAGIC = -2,
	/** The image uses an unsupported format version. */
	MODEL_IMAGE_ERR_BAD_FORMAT_VERSION = -3,
	/** The image size is smaller than the image header or does not fit the partition. */
	MODEL_IMAGE_ERR_BAD_SIZE = -4,
	/** The image is for a different backend than the loader. */
	MODEL_IMAGE_ERR_BAD_BACKEND = -5,
	/** The image was built against a different firmware contract. */
	MODEL_IMAGE_ERR_CONTRACT_MISMATCH = -6,
	/** The model needs more persistent vars than the firmware provides. */
	MODEL_IMAGE_ERR_PERSISTENT_VARS_TOO_MANY = -7,
	/** The model needs a larger packed-output buffer than the firmware provides. */
	MODEL_IMAGE_ERR_PACKED_OUTPUT_TOO_LARGE = -8,
	/** The image was linked against firmware symbols the firmware does not provide. */
	MODEL_IMAGE_ERR_BINDING_MISMATCH = -9,
	/** The Axon model is missing or failed validation. */
	MODEL_IMAGE_ERR_AXON_VALIDATE = -10,
};

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_IMAGE_RESULT_H_ */

/**
 * @}
 */
