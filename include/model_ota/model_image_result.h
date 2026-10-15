/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_IMAGE_RESULT_H_
#define MODEL_OTA_MODEL_IMAGE_RESULT_H_

/**
 * @file
 * @brief Return codes for OTA-wired model partition loaders.
 *
 * Full image format and loader APIs live in the model OTA library
 * (lib/model_ota/include/model_ota/model_image.h), not in the module include tree.
 */

#ifdef __cplusplus
extern "C" {
#endif

/** Return codes for wired @c nrf_edgeai_load_user_model_* and @c model_ota_load_axon_* loaders. */
enum model_image_result {
	MODEL_IMAGE_OK = 0,
	MODEL_IMAGE_ERR_NO_PARTITION = -1,
	MODEL_IMAGE_ERR_FLASH_READ = -2,
	MODEL_IMAGE_ERR_BAD_MAGIC = -3,
	MODEL_IMAGE_ERR_BAD_FORMAT_VERSION = -4,
	MODEL_IMAGE_ERR_TOO_LARGE = -5,
	MODEL_IMAGE_ERR_BAD_CRC = -6,
	MODEL_IMAGE_ERR_NEURONS_BUF_TOO_SMALL = -8,
	MODEL_IMAGE_ERR_PARAMS_TYPE_MISMATCH = -10,
	MODEL_IMAGE_ERR_NOT_AXON_IMAGE = -13,
	MODEL_IMAGE_ERR_AXON_VALIDATE = -14,
	MODEL_IMAGE_ERR_BAD_PARAMS_TYPE = -15,
	MODEL_IMAGE_ERR_CONTRACT_MISMATCH = -16,
	MODEL_IMAGE_ERR_BINDING_MISMATCH = -18,
	MODEL_IMAGE_ERR_PERSISTENT_VARS_TOO_MANY = -19,
	MODEL_IMAGE_ERR_PACKED_OUTPUT_TOO_LARGE = -20,
	MODEL_IMAGE_ERR_DSP_MASK_MISMATCH = -22,
	MODEL_IMAGE_ERR_UPDATE_IN_PROGRESS = -23,
};

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_IMAGE_RESULT_H_ */
