/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/util.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>

#include "hello_axon_model_image.h"
#include "model_partition.h"

BUILD_ASSERT(PARTITION_EXISTS(axon_model_partition));

static NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_runtime[HELLO_AXON_MODEL_CMD_BUFFER_LEN];

static bool header_is_valid(const struct hello_axon_model_image_header *header)
{
	return header->magic == HELLO_AXON_MODEL_IMAGE_MAGIC &&
	       header->version == HELLO_AXON_MODEL_IMAGE_VERSION &&
	       header->model_const_size == HELLO_AXON_MODEL_CONST_SIZE &&
	       header->cmd_buffer_len == HELLO_AXON_MODEL_CMD_BUFFER_LEN &&
	       header->metadata_offset == sizeof(*header) &&
	       header->model_const_offset ==
		       header->metadata_offset + sizeof(struct hello_axon_model_image_metadata) &&
	       header->cmd_buffer_offset == header->model_const_offset + HELLO_AXON_MODEL_CONST_SIZE &&
	       header->model_name_offset ==
		       header->cmd_buffer_offset + (HELLO_AXON_MODEL_CMD_BUFFER_LEN * sizeof(uint32_t));
}

static void build_runtime_cmd_buffer(const uint8_t *partition_base,
				     const struct hello_axon_model_image_header *header,
				     const struct hello_axon_model_const_layout *model_const)
{
	const uint32_t *cmd_template =
		(const uint32_t *)(partition_base + header->cmd_buffer_offset);
	const uintptr_t const_base = (uintptr_t)model_const;

	for (size_t i = 0; i < header->cmd_buffer_len; i++) {
		const uint32_t word = cmd_template[i];

		if (word == HELLO_AXON_CMD_PTR_INTERLAYER) {
			cmd_buffer_runtime[i] =
				(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)(uintptr_t)
					nrf_axon_interlayer_buffer;
		} else if ((word & 0xF0000000U) == 0x80000000U) {
			cmd_buffer_runtime[i] =
				(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)(const_base +
									 (word & 0x0FFFFFFFU));
		} else {
			cmd_buffer_runtime[i] = word;
		}
	}
}

int hello_axon_model_partition_load(nrf_axon_nn_compiled_model_s *model)
{
	const uint8_t *partition_base = UINT_TO_POINTER(PARTITION_ADDRESS(axon_model_partition));
	const struct hello_axon_model_image_header *header =
		(const struct hello_axon_model_image_header *)partition_base;
	const struct hello_axon_model_image_metadata *metadata;
	const struct hello_axon_model_const_layout *model_const;
	const char *model_name;

	if (!header_is_valid(header)) {
		return -EINVAL;
	}

	metadata = (const struct hello_axon_model_image_metadata *)(partition_base +
								     header->metadata_offset);
	model_const = (const struct hello_axon_model_const_layout *)(partition_base +
								     header->model_const_offset);
	model_name = (const char *)(partition_base + header->model_name_offset);

	if (strnlen(model_name, HELLO_AXON_MODEL_NAME_MAX_LEN) >= HELLO_AXON_MODEL_NAME_MAX_LEN) {
		return -EINVAL;
	}

	build_runtime_cmd_buffer(partition_base, header, model_const);

	*model = (nrf_axon_nn_compiled_model_s){
		.compiler_version = metadata->compiler_version,
		.model_name = model_name,
		.labels = NULL,
		.inputs = {
			{
				.ptr = (int8_t *)nrf_axon_interlayer_buffer,
				.dimensions = {
					.height = metadata->input_height,
					.width = metadata->input_width,
					.channel_cnt = metadata->input_channel_cnt,
					.byte_width = metadata->input_byte_width,
				},
				.quant_mult = metadata->input_quant_mult,
				.stride = metadata->input_stride,
				.quant_round = metadata->input_quant_round,
				.quant_zp = metadata->input_quant_zp,
				.is_external = metadata->is_external != 0,
			},
		},
		.input_cnt = metadata->input_cnt,
		.external_input_ndx = metadata->external_input_ndx,
		.output_ptr = (int8_t *)nrf_axon_interlayer_buffer,
		.packed_output_buf = NULL,
		.interlayer_buffer_needed = metadata->interlayer_buffer_needed,
		.psum_buffer_needed = metadata->psum_buffer_needed,
		.cmd_buffer_ptr = cmd_buffer_runtime,
		.model_const_ptr = model_const,
		.model_const_size = sizeof(*model_const),
		.cmd_buffer_len = header->cmd_buffer_len,
		.persistent_vars = {
			.count = 0,
		},
		.output_dimensions = {
			.height = metadata->output_height,
			.width = metadata->output_width,
			.channel_cnt = metadata->output_channel_cnt,
			.byte_width = metadata->output_byte_width,
		},
		.output_dequant_mult = metadata->output_dequant_mult,
		.output_dequant_round = metadata->output_dequant_round,
		.output_dequant_zp = metadata->output_dequant_zp,
		.output_stride = metadata->output_stride,
		.is_layer_model = metadata->is_layer_model != 0,
		.extra_output_cnt = metadata->extra_output_cnt,
		.extra_outputs = NULL,
		.min_driver_version_required = metadata->min_driver_version_required,
	};

	return 0;
}
