/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include <zephyr/devicetree.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/util.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>

#include "hello_axon_model_image.h"
#include "model_partition.h"

BUILD_ASSERT(PARTITION_EXISTS(axon_model_partition));

struct hello_axon_model_const_layout {
	int8_t l00_weights[16];
	int32_t l00_biasp[16];
	int8_t l01_weights[256];
	int32_t l01_biasp[16];
	int8_t l02_weights[16];
	int32_t l02_biasp[1];
};

BUILD_ASSERT(sizeof(struct hello_axon_model_const_layout) == HELLO_AXON_MODEL_CONST_SIZE);

struct cmd_buffer_fixup {
	uint8_t word_index;
	uint16_t const_offset;
};

static NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE cmd_buffer_runtime[69];

static const uint32_t cmd_buffer_template[69] = {
	0x1fff0044U,
	0x02000080U, 0x00010001U, 0U, 0x00330001U,
	0x02000090U, 0x00100001U, 0U, 0x00330001U,
	0x050000a0U, 0x00010010U, 0U, 0x00050040U, 0x00010010U, 0U, 0x00030010U,
	0x000000bcU, 0x00000000U,
	0x040000c8U, 0x00000000U, 0x00000000U, 0x03010000U, 0x00011703U, 0x00000205U,
	0x01000180U, 0x00000003U, 0x00000000U,
	0x010001a4U, 0x00000000U, 0x000116d5U,
	0x010001c8U, 0xc0000000U, 0xc0000000U,
	0x000000f0U, 0x00000100U,
	0x00000080U, 0x00010010U,
	0x00000088U, 0x00330010U,
	0x01000090U, 0x00100010U, 0U,
	0x000000a4U, 0U,
	0x000001a8U, 0x00035a4eU,
	0x000000f0U, 0x00000100U,
	0x01000090U, 0x00010010U, 0U,
	0x030000a0U, 0x00010001U, 0U, 0x00050004U, 0x00010001U,
	0x000000b4U, 0x00030001U,
	0x000000d4U, 0x00011c03U,
	0x01000180U, 0x00000002U, 0x80000000U,
	0x000001a8U, 0x00183066U,
	0x000001ccU, 0x40000000U,
	0x000000f0U, 0x00000100U,
};

static const struct cmd_buffer_fixup cmd_buffer_const_fixups[] = {
	{ .word_index = 7, .const_offset = offsetof(struct hello_axon_model_const_layout, l00_weights) },
	{ .word_index = 11, .const_offset = offsetof(struct hello_axon_model_const_layout, l00_biasp) },
	{ .word_index = 41, .const_offset = offsetof(struct hello_axon_model_const_layout, l01_weights) },
	{ .word_index = 43, .const_offset = offsetof(struct hello_axon_model_const_layout, l01_biasp) },
	{ .word_index = 50, .const_offset = offsetof(struct hello_axon_model_const_layout, l02_weights) },
	{ .word_index = 53, .const_offset = offsetof(struct hello_axon_model_const_layout, l02_biasp) },
};

static const uint8_t cmd_buffer_interlayer_fixups[] = { 3, 14 };

static void build_runtime_cmd_buffer(const struct hello_axon_model_const_layout *model_const)
{
	const uintptr_t const_base = (uintptr_t)model_const;

	for (size_t i = 0; i < ARRAY_SIZE(cmd_buffer_template); i++) {
		cmd_buffer_runtime[i] = cmd_buffer_template[i];
	}

	for (size_t i = 0; i < ARRAY_SIZE(cmd_buffer_interlayer_fixups); i++) {
		const uint8_t index = cmd_buffer_interlayer_fixups[i];

		cmd_buffer_runtime[index] =
			(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)(uintptr_t)nrf_axon_interlayer_buffer;
	}

	for (size_t i = 0; i < ARRAY_SIZE(cmd_buffer_const_fixups); i++) {
		const struct cmd_buffer_fixup *fixup = &cmd_buffer_const_fixups[i];

		cmd_buffer_runtime[fixup->word_index] =
			(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE)(const_base + fixup->const_offset);
	}
}

int hello_axon_model_partition_load(nrf_axon_nn_compiled_model_s *model)
{
	const uint8_t *partition_base = UINT_TO_POINTER(PARTITION_ADDRESS(axon_model_partition));
	const struct hello_axon_model_image_header *header =
		(const struct hello_axon_model_image_header *)partition_base;
	const struct hello_axon_model_const_layout *model_const;

	if (header->magic != HELLO_AXON_MODEL_IMAGE_MAGIC) {
		return -EINVAL;
	}

	if (header->version != HELLO_AXON_MODEL_IMAGE_VERSION) {
		return -EINVAL;
	}

	if (header->model_const_size != HELLO_AXON_MODEL_CONST_SIZE) {
		return -EINVAL;
	}

	model_const = (const struct hello_axon_model_const_layout *)(partition_base + sizeof(*header));
	build_runtime_cmd_buffer(model_const);

	*model = (nrf_axon_nn_compiled_model_s){
		.compiler_version = 0x00010201,
		.model_name = "hello_axon",
		.labels = NULL,
		.inputs = {
			{
				.ptr = (int8_t *)nrf_axon_interlayer_buffer,
				.dimensions = {
					.height = 1,
					.width = 1,
					.channel_cnt = 1,
					.byte_width = 1,
				},
				.quant_mult = 21335090,
				.stride = 1,
				.quant_round = 19,
				.quant_zp = -128,
				.is_external = true,
			},
		},
		.input_cnt = 1,
		.external_input_ndx = 0,
		.output_ptr = (int8_t *)nrf_axon_interlayer_buffer,
		.packed_output_buf = NULL,
		.interlayer_buffer_needed = NRF_AXON_MODEL_HELLO_AXON_MAX_IL_BUFFER_USED,
		.psum_buffer_needed = NRF_AXON_MODEL_HELLO_AXON_MAX_PSUM_BUFFER_USED,
		.cmd_buffer_ptr = cmd_buffer_runtime,
		.model_const_ptr = model_const,
		.model_const_size = sizeof(*model_const),
		.cmd_buffer_len = ARRAY_SIZE(cmd_buffer_runtime),
		.persistent_vars = {
			.count = 0,
		},
		.output_dimensions = {
			.height = 1,
			.width = 1,
			.channel_cnt = 1,
			.byte_width = 1,
		},
		.output_dequant_mult = 4548375,
		.output_dequant_round = 29,
		.output_dequant_zp = 4,
		.output_stride = 4,
		.is_layer_model = false,
		.extra_output_cnt = 0,
		.extra_outputs = NULL,
		.min_driver_version_required = 0x00010200,
	};

	return 0;
}
