/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Linker script for the standalone model partition image.
 *
 * The image is linked at NRF_AXON_MODEL_PARTITION_ADDR (from devicetree). All
 * rodata from the included model header lands in .model_image after the
 * partition header placed by model_image_stub.c.
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
