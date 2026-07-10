/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <stdint.h>

#define NRF_AXON_MODEL_PARTITION_MAGIC   0x4E4F5841U /* 'AXON' little-endian */
#define NRF_AXON_MODEL_PARTITION_VERSION 4U

struct nrf_axon_model_partition_header {
	uint32_t magic;
	uint32_t version;
	uint32_t model_offset;
	uint32_t image_size;
};
