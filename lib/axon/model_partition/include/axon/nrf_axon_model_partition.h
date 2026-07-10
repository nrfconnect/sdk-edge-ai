/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <stdbool.h>
#include <stdint.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#include <axon/nrf_axon_model_partition_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Validate an Axon model partition image header.
 *
 * Checks magic, version, image_size bounds, model_offset placement, and that
 * the image is large enough to hold at least a compiled model struct.
 *
 * @param base_addr CPU address where the partition is mapped.
 * @retval true Image header is valid.
 * @retval false Image header is invalid.
 */
bool nrf_axon_model_partition_is_valid(uintptr_t base_addr);

/**
 * @brief Return the compiled model stored in a partition image.
 *
 * The returned structure resides in the partition (NVM-mapped). All pointer
 * fields are resolved at model-image link time.
 *
 * @param base_addr CPU address where the partition is mapped.
 * @retval Pointer to the model, or NULL if the header is invalid.
 */
const nrf_axon_nn_compiled_model_s *nrf_axon_model_partition_get(uintptr_t base_addr);

#ifdef __cplusplus
}
#endif
