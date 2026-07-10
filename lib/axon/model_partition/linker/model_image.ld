/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

SECTIONS
{
	. = NRF_AXON_MODEL_PARTITION_ADDR;
	.model_image : {
		__axon_model_image_start = .;
		KEEP(*(.model_image.partition_hdr))
		KEEP(*(.rodata))
		KEEP(*(.rodata.*))
		KEEP(*(.model_image))
		__axon_model_image_end = .;
	} = 0
}
