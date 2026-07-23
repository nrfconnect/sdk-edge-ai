/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_IMAGE_COMMON_H_
#define MODEL_IMAGE_COMMON_H_

#include <model_ota/model_image.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Read the partition header and run checks shared by all backends.
 *
 * Opens the flash area, reads @ref model_image_header, validates magic,
 * @ref model_image_header.format_version, and image size against the partition,
 * closes the flash area, then verifies CRC32/IEEE over the memory-mapped image.
 *
 * @param[in]  fa_id           Flash area ID of the partition.
 * @param[in]  partition_addr  Memory-mapped base address of that partition.
 * @param[out] hdr_out         Validated header copy on success.
 * @retval MODEL_IMAGE_OK (0) on success, a negative @ref model_image_result otherwise.
 */
int model_image_read_and_validate(uint8_t fa_id, const uint8_t *partition_addr,
				  struct model_image_header *hdr_out);

/**
 * @brief True iff [p, p + nbytes) lies fully inside [base, end).
 *
 * A NULL @p is never in range. The (p + nbytes) >= p guard rejects a span that
 * would wrap the address space.
 */
bool model_image_span_in_image(const void *p, size_t nbytes, const uint8_t *base,
			       const uint8_t *end);

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
