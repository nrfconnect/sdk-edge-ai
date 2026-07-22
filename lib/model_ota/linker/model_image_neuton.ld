/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Linker script for a standalone Neuton model partition image.
 *
 * The image is linked at NRF_MODEL_PARTITION_ADDR (supplied via --defsym from the partition's
 * devicetree reg address). The header stub is placed first, at the base, followed by all
 * .rodata that survives --gc-sections - i.e. the compile-time descriptor `model_instance_` and
 * the data/scale arrays it (and the header) reference. Everything unreferenced (the runtime
 * nrf_edgeai_t object with its pointers to application code, the DSP pipeline, input scales, ...)
 * is garbage-collected, which is also what keeps the link free of undefined app symbols.
 *
 * Only the header is KEEP()'d: it is the sole GC root, and every byte in the image is reachable
 * from it. Because the whole image is placed at the partition base, all intra-image pointers are
 * correct absolute flash addresses.
 */

OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

SECTIONS
{
	. = NRF_MODEL_PARTITION_ADDR;
	.model_image : {
		__model_image_start = .;
		KEEP(*(.model_image.header))
		*(.rodata)
		*(.rodata.*)
		__model_image_end = .;
	} = 0
}
