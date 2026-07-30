/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Linker script for a standalone model partition image (Neuton or Axon).
 *
 * The image is linked at MODEL_IMAGE_LINK_ADDR (supplied via --defsym from
 * model_ota_image_link_addr(); partition base + MCUboot header offset when applicable).
 * The header stub is placed first, followed by all .rodata that survives --gc-sections -
 * i.e. the compile-time model descriptor and the data
 * arrays it (and the header) reference. Everything unreferenced is garbage-collected, which
 * is also what keeps the Neuton link free of undefined app symbols; Axon app-owned RAM pointers
 * are resolved separately via a generated PROVIDE() linker fragment.
 *
 * Only the header is KEEP()'d: it is the sole GC root, and every byte in the image is reachable
 * from it. Because the whole image is placed at MODEL_IMAGE_LINK_ADDR, all intra-image
 * pointers are correct absolute flash addresses.
 */

OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

SECTIONS
{
	. = MODEL_IMAGE_LINK_ADDR;
	.model_image : {
		__model_image_start = .;
		KEEP(*(.model_image.header))
		*(.rodata)
		*(.rodata.*)
		__model_image_end = .;
	} = 0
}
