/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <stdint.h>

#define HELLO_AXON_MODEL_IMAGE_MAGIC   0x4E4F5841U /* 'AXON' little-endian */
#define HELLO_AXON_MODEL_IMAGE_VERSION 1U
#define HELLO_AXON_MODEL_CONST_SIZE    420U

#define NRF_AXON_MODEL_HELLO_AXON_MAX_IL_BUFFER_USED   16
#define NRF_AXON_MODEL_HELLO_AXON_MAX_PSUM_BUFFER_USED 0

struct hello_axon_model_image_header {
	uint32_t magic;
	uint32_t version;
	uint32_t model_const_size;
	uint32_t reserved;
};

BUILD_ASSERT(sizeof(struct hello_axon_model_image_header) == 16);
