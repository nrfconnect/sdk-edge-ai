/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MODEL_OTA_MODEL_IMAGE_COMMON_H_
#define MODEL_OTA_MODEL_IMAGE_COMMON_H_

#include <stddef.h>
#include <stdint.h>

#include <model_ota/model_image_result.h>

#include "model_image.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run the checks shared by all backends on a memory-mapped model image.
 *
 * Checks the magic, format version and image size against @p max_image_size.
 *
 * @param[in]  image_addr     Memory-mapped address of the model image in the model partition.
 * @param[in]  max_image_size Bytes available from @p image_addr to the end of the model partition.
 * @param[out] hdr_out        Set to the image header on success.
 * @retval MODEL_IMAGE_OK on success, a negative @ref model_image_result otherwise.
 */
enum model_image_result model_image_validate_header(const uint8_t *image_addr,
						    size_t max_image_size,
						    const struct model_image_header **hdr_out);

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_IMAGE_COMMON_H_ */
