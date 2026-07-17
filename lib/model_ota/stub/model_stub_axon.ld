/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Linker script for the standalone Axon model stub link (see ../cmake/nrf_axon_model_stub.cmake).
 *
 * .model_stub_header_reserve is a fixed-size placeholder (see model_stub_axon.c) exactly the
 * size of package_model_axon.py's on-flash header struct, placed right before everything else
 * so that .model_stub - the model header's own data, and therefore package_base - starts
 * exactly where the payload will actually sit once that real header is flashed ahead of it.
 * package_model_axon.py only ever overwrites those placeholder bytes; it never shifts anything
 * that follows.
 *
 * Everything the compiled model header defines (the compiled model struct, cmd_buffer,
 * model_const, and optionally extra_outputs/labels) is const, so it all lands in .rodata here.
 * Nothing in the stub should ever need .bss/.data: every field that would otherwise require
 * writable app RAM (nrf_axon_interlayer_buffer, persistent_vars arrays) is declared extern in
 * the patched header this is linked against and resolved via a generated PROVIDE() script
 * instead - see tools/model_ota/gen_axon_stub_fixups.py and extract_elf_syms.py.
 */
OUTPUT_FORMAT("elf32-littlearm", "elf32-littlearm", "elf32-littlearm")
OUTPUT_ARCH(arm)

SECTIONS
{
	. = MODEL_STUB_ADDR;
	.model_stub_header_reserve : {
		KEEP(*(.model_stub_header_reserve))
	}
	.model_stub : {
		KEEP(*(.rodata))
		KEEP(*(.rodata.*))
		KEEP(*(.data))
		KEEP(*(.data.*))
	} = 0
}
