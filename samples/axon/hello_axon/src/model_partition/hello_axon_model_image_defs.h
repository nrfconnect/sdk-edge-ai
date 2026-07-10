/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <stdint.h>

#define HELLO_AXON_MODEL_IMAGE_MAGIC   0x4E4F5841U /* 'AXON' little-endian */
#define HELLO_AXON_MODEL_IMAGE_VERSION 3U
#define HELLO_AXON_MODEL_CONST_SIZE    420U
#define HELLO_AXON_MODEL_CMD_BUFFER_LEN  69U
#define HELLO_AXON_MODEL_NAME_MAX_LEN    32U

struct hello_axon_model_image_header {
	uint32_t magic;
	uint32_t version;
	uint32_t model_const_size;
	uint32_t cmd_buffer_len;
	uint32_t metadata_offset;
	uint32_t model_const_offset;
	uint32_t cmd_buffer_offset;
	uint32_t model_name_offset;
};

struct hello_axon_model_image_metadata {
	uint32_t compiler_version;
	uint32_t interlayer_buffer_needed;
	uint32_t psum_buffer_needed;
	uint32_t min_driver_version_required;
	uint32_t output_dequant_mult;
	uint32_t input_quant_mult;
	uint16_t input_height;
	uint16_t input_width;
	uint16_t input_channel_cnt;
	uint16_t output_height;
	uint16_t output_width;
	uint16_t output_channel_cnt;
	uint8_t input_byte_width;
	uint8_t input_quant_round;
	int8_t input_quant_zp;
	uint16_t input_stride;
	uint8_t output_byte_width;
	uint8_t output_dequant_round;
	int8_t output_dequant_zp;
	uint16_t output_stride;
	uint8_t input_cnt;
	int8_t external_input_ndx;
	uint8_t is_external;
	uint8_t is_layer_model;
	uint16_t extra_output_cnt;
} __attribute__((__packed__));

struct hello_axon_model_const_layout {
	int8_t l00_weights[16];
	int32_t l00_biasp[16];
	int8_t l01_weights[256];
	int32_t l01_biasp[16];
	int8_t l02_weights[16];
	int32_t l02_biasp[1];
};

struct hello_axon_model_image_blob {
	struct hello_axon_model_image_header header;
	struct hello_axon_model_image_metadata metadata;
	struct hello_axon_model_const_layout model_const;
	uint32_t cmd_buffer[HELLO_AXON_MODEL_CMD_BUFFER_LEN];
	char model_name[HELLO_AXON_MODEL_NAME_MAX_LEN];
};
