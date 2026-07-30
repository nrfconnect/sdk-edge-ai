/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_IMAGE_COMMON_H_
#define MODEL_IMAGE_COMMON_H_

#include <model_ota/model_image.h>

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Byte offset from a model partition base to the linked model image.
 *
 * Each partition reserves an MCUboot image-header slot at its base; the model payload is
 * linked immediately after that slot (@c IMAGE_HEADER_SIZE bytes).
 */
size_t model_image_partition_payload_offset(void);

/**
 * @brief Read the partition header and run checks shared by all backends.
 *
 * Reads @ref model_image_header directly from the memory-mapped (XIP) partition,
 * validates magic, @ref model_image_header.format_version, and image size against
 * @p partition_size, then verifies CRC32/IEEE over the whole mapped image.
 *
 * @param[in]  partition_addr      Memory-mapped base address of a zephyr,mapped-partition node.
 * @param[in]  partition_size      Size of that partition, in bytes.
 * @param[out] hdr_out             Validated header copy on success.
 * @retval MODEL_IMAGE_OK (0) on success, a negative @ref model_image_result otherwise.
 */
int model_image_read_and_validate(const uint8_t *partition_addr, size_t partition_size,
				  struct model_image_header *hdr_out);

/**
 * @brief Weight/neuron element size for a Neuton @ref model_image_params_type.
 *
 * @param[in]  params_type   One of MODEL_IMAGE_PARAMS_F32/Q16/Q8.
 * @param[out] elem_size_out Set to the element size in bytes on success, or NULL to validate
 *                           @p params_type only.
 * @retval MODEL_IMAGE_OK on success, @ref MODEL_IMAGE_ERR_BAD_PARAMS_TYPE otherwise.
 */
int model_image_neuton_params_elem_size(uint8_t params_type, size_t *elem_size_out);

#endif /* MODEL_IMAGE_COMMON_H_ */
