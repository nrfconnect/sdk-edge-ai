#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side packaging tool for the model-only OTA update PoC (Axon).

This tool captures a model's *entire* compiler-generated nrf_axon_nn_compiled_model_s struct
verbatim (see axon_struct_layout.py for the ctypes mirror of its ABI) and relocates every
pointer field inside it that refers to flash-owned model data (cmd_buffer, model_const,
extra_outputs, and cmd_buffer's own internal pointers into model_const) from wherever the
reference build happened to link them to wherever they will actually live once flashed into the
model_storage partition (--address). Pointer fields that refer to app-owned RAM
(nrf_axon_interlayer_buffer) are instead reduced to a byte offset into that buffer, since the
buffer's actual address is only known to the deployed firmware, not to this tool. That means the
on-device loader (model_pkg_load_axon()) never has to patch anything shape-specific at runtime -
it copies the struct straight out of the memory-mapped partition (true XIP, zero RAM cost for
everything except that one fixed-size struct copy) and only fixes up the handful of RAM-owned
fields. Packaging therefore needs two inputs:

1. A "reference build" of a sample (e.g. samples/axon/hello_axon) with its
   *_REFERENCE_BUILD=y Kconfig option set - never flashed, its only purpose is to link in the
   generated model header just enough to fix real addresses for model_<name>/
   cmd_buffer_<name>/axon_model_const_<name> in its zephyr.elf. Build it with, e.g.:

       west build -b nrf54lm20dk/nrf54lm20b/cpuapp -d build_ref \\
           samples/axon/hello_axon -- -DCONFIG_HELLO_AXON_REFERENCE_BUILD=y

2. Nothing else - unlike prior versions of this tool, no separate generated-header text file is
   parsed: every scalar value (quantization, buffer sizing, driver/compiler version, ...) is
   already inside the struct captured from the ELF, since it is a byte-for-byte copy of the
   compiler's own output.

Usage:
    python3 package_model_axon.py \\
        --elf build_ref/zephyr/zephyr.elf \\
        --model-name hello_axon -o model_v1 --dts build_ref/zephyr/zephyr.dts

--dts reads the model_storage partition's actual address and size straight out of a build's
generated zephyr.dts, so a stale/wrong --address can't silently produce a package with
mis-relocated pointers, or one that overflows the partition. Falls back to --address (default
0x102000) and --partition-size (default 968 KiB) - the nRF54LM20 DK values - if omitted.

Produces model_v1.bin (raw package) and model_v1.hex (same bytes, addressed for flashing
directly into the model_storage partition, independently of the application image), e.g.:

    nrfutil device program --firmware model_v1.hex \\
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

Requires pyelftools (already bundled with the NCS toolchain's Python; otherwise
`pip install pyelftools`).

Known limitations (see the plan this tool was implemented from for the reasoning): models using
`labels` (per-class text labels) or `persistent_vars` (streaming/VarHandle models) are rejected
outright - resolving an arbitrary string literal's address reliably from an ELF, and computing
per-persistent-variable buffer offsets, were judged not worth the added complexity for a PoC
with no such model to validate against yet. Multi-input models (input_cnt > 1, both external)
and models with extra_outputs are supported by this tool and the loader, but - absent such a
model in this repo - are only exercised by this tool's own unit tests, not on real hardware.
"""
import argparse
import ctypes
import struct
import sys
import zlib
from pathlib import Path

from elftools.elf.elffile import ELFFile

from axon_struct_layout import CompiledModel, OutputDesc, check_struct_size
from model_partition_layout import (
	DEFAULT_ADDRESS,
	DEFAULT_PARTITION_SIZE,
	check_package_fits,
	read_partition_layout_from_dts,
	report_package_usage,
)

MAGIC = b"NEAI"
FORMAT_VERSION = 3
MODEL_TYPE_AXON = 1
NAME_LEN = 16

# <magic(4s) format_version(H) model_type(B) reserved0(B) name(16s) model_version(I)
#  payload_size(I) section_len[4](4I) struct_size(I) package_base(I) crc32(I) -- packed,
#  little-endian, no padding.
HEADER_FMT = "<4sHBB16sII4III" "I"
HEADER_LEN = struct.calcsize(HEADER_FMT)


def parse_version(v):
	"""Accepts "1.2.3" or an already-numeric value; packs as (major<<16)|(minor<<8)|patch."""
	if isinstance(v, int):
		return v
	parts = [int(p) for p in str(v).split(".")]
	while len(parts) < 3:
		parts.append(0)
	major, minor, patch = parts[:3]
	return (major << 16) | (minor << 8) | patch


class ElfSymbols:
	"""Thin wrapper around an ELF's .symtab for the two lookups this tool needs: by name (for
	symbols whose generated name we know, e.g. model_<name>) and by address (for pointer
	fields whose target's name isn't predictable, e.g. extra_outputs)."""

	def __init__(self, elf_path):
		self.elf_path = elf_path
		with open(elf_path, "rb") as f:
			elffile = ELFFile(f)
			symtab = elffile.get_section_by_name(".symtab")
			if symtab is None:
				raise ValueError(
					"%s has no .symtab (build without stripping symbols)" % elf_path)
			# Only initialized-data symbols (st_size > 0) in a regular section are of
			# interest here; this also filters out SHN_UNDEF/SHN_ABS/... symbols whose
			# st_shndx isn't a plain section index.
			self.symbols = [s for s in symtab.iter_symbols() if s["st_size"] > 0]
			# Cache raw section bytes now, while the file is still open.
			self._section_cache = {}
			for sym in self.symbols:
				shndx = sym["st_shndx"]
				if not isinstance(shndx, int) or shndx in self._section_cache:
					continue
				section = elffile.get_section(shndx)
				if section is not None:
					self._section_cache[shndx] = (section["sh_addr"], section.data(),
								       section.name)

	def by_name(self, name):
		"""Returns (address, raw_bytes) for a named symbol's initialized data."""
		matches = [s for s in self.symbols if s.name == name]
		if not matches:
			raise ValueError(
				"Symbol '%s' not found in %s - was the reference build's generated "
				"model header actually linked in?" % (name, self.elf_path))
		return self._read(matches[0])

	def by_address(self, address):
		"""Returns (symbol_name, raw_bytes) for whichever symbol's [address, address+size)
		range contains `address`. Used for pointer fields (e.g. extra_outputs) whose
		target symbol name is not predictable from the model name alone."""
		for sym in self.symbols:
			if sym["st_value"] <= address < sym["st_value"] + sym["st_size"]:
				_, data = self._read(sym)
				return sym.name, data
		raise ValueError(
			"No symbol in %s covers address 0x%08x - cannot resolve what this pointer "
			"field targets" % (self.elf_path, address))

	def _read(self, sym):
		address = sym["st_value"]
		size = sym["st_size"]
		shndx = sym["st_shndx"]
		if not isinstance(shndx, int) or shndx not in self._section_cache:
			raise ValueError("Symbol '%s' is not defined in a regular section (shndx=%r)" %
					  (sym.name, shndx))
		sh_addr, data, section_name = self._section_cache[shndx]
		offset = address - sh_addr
		chunk = data[offset:offset + size]
		if len(chunk) != size:
			raise ValueError("Could not read %u bytes for symbol '%s' from section '%s'" %
					  (size, sym.name, section_name))
		return address, chunk


def validate_shape(model, model_name):
	"""Raises ValueError for known-unsupported model shapes (see module docstring). Pure
	function - no ELF access - so it can be unit-tested directly against synthetic
	CompiledModel instances built from raw bytes."""
	if model.labels != 0:
		raise ValueError(
			"model_%s has a non-NULL 'labels' pointer - packaging models with text "
			"labels is not supported yet" % model_name)
	if model.persistent_vars.count != 0:
		raise ValueError(
			"model_%s has %u persistent variable(s) - packaging streaming/VarHandle "
			"models is not supported yet" % (model_name, model.persistent_vars.count))
	if model.packed_output_buf != 0:
		raise ValueError(
			"model_%s has a non-NULL 'packed_output_buf' - dedicated packed-output "
			"buffers are not supported (they are compile-time arrays the deployed "
			"app, which no longer links in the generated model header, cannot "
			"provide)" % model_name)
	if model.input_cnt == 0 or model.input_cnt > len(model.inputs):
		raise ValueError("model_%s has an implausible input_cnt=%u" %
				  (model_name, model.input_cnt))
	for i in range(model.input_cnt):
		if not model.inputs[i].is_external:
			raise ValueError(
				"model_%s.inputs[%u] is not external - internal inputs (fed by "
				"another layer or a persistent variable) are not supported yet" %
				(model_name, i))
	if model.extra_output_cnt > 0 and model.extra_outputs == 0:
		raise ValueError(
			"model_%s has extra_output_cnt=%u but a NULL extra_outputs pointer" %
			(model_name, model.extra_output_cnt))
	if model.extra_output_cnt == 0 and model.extra_outputs != 0:
		raise ValueError(
			"model_%s has extra_output_cnt=0 but a non-NULL extra_outputs pointer" %
			model_name)


def relocate_ram_pointers(model, model_name, interlayer_addr):
	"""Rewrites model's RAM-owned pointer fields (inputs[i].ptr for external inputs,
	output_ptr) from absolute reference-build addresses into nrf_axon_interlayer_buffer to
	byte offsets - the on-device loader adds those offsets to whatever address it actually
	has for that buffer, since it isn't necessarily the reference build's - and clears
	model_name (the loader always repoints it at the package's own name field instead, see
	model_pkg_axon.c). Call validate_shape() first. Pure function - no ELF access - so it can
	be unit-tested directly against synthetic CompiledModel instances built from raw bytes."""
	def to_offset(field_name, value):
		offset = (value - interlayer_addr) & 0xFFFFFFFF
		if offset >= 0x10000000:
			raise ValueError(
				"model_%s.%s (0x%08x) does not look like it points into "
				"nrf_axon_interlayer_buffer (0x%08x) - computed offset 0x%08x is "
				"suspiciously large" % (model_name, field_name, value,
							 interlayer_addr, offset))
		return offset

	for i in range(model.input_cnt):
		model.inputs[i].ptr = to_offset("inputs[%u].ptr" % i, model.inputs[i].ptr)
	model.output_ptr = to_offset("output_ptr", model.output_ptr)
	model.model_name = 0


def relocate_flash_pointers(model, cmd_buffer_base, model_const_base, extra_outputs_base):
	"""Rewrites model's flash-owned pointer fields to their final address once flashed into
	model_storage. Pure function - no ELF access - so it can be unit-tested directly against
	synthetic CompiledModel instances built from raw bytes."""
	model.cmd_buffer_ptr = cmd_buffer_base
	model.model_const_ptr = model_const_base
	if model.extra_output_cnt > 0:
		model.extra_outputs = extra_outputs_base


def relocate_cmd_buffer(cmd_buffer_bytes, old_base, new_base, span):
	"""Rewrite every 32-bit word in cmd_buffer_bytes that falls in [old_base, old_base+span)
	by adding (new_base - old_base), leaving every other word untouched. This is the same
	relocation the on-device loader used to perform at every boot; doing it once here means
	the loader can wire cmd_buffer straight into flash with no runtime patching at all."""
	words = list(struct.unpack("<%uI" % (len(cmd_buffer_bytes) // 4), cmd_buffer_bytes))
	delta = (new_base - old_base) & 0xFFFFFFFF
	patched = 0
	for i, word in enumerate(words):
		if old_base <= word < old_base + span:
			words[i] = (word + delta) & 0xFFFFFFFF
			patched += 1
	print("Relocated %u/%u cmd_buffer words from model_const at 0x%08x to 0x%08x" %
	      (patched, len(words), old_base, new_base))
	return struct.pack("<%uI" % len(words), *words)


def build_package(elf_path, model_name, version, address):
	syms = ElfSymbols(elf_path)

	_struct_addr, struct_bytes = syms.by_name("model_%s" % model_name)
	check_struct_size(len(struct_bytes), elf_path)

	buf = bytearray(struct_bytes)
	model = CompiledModel.from_buffer(buf)

	validate_shape(model, model_name)

	ref_const_base, const_bytes = syms.by_name("axon_model_const_%s" % model_name)
	if model.model_const_ptr != ref_const_base or model.model_const_size != len(const_bytes):
		raise ValueError(
			"model_%s.model_const_ptr/model_const_size (0x%08x, %u B) do not match "
			"symbol axon_model_const_%s (0x%08x, %u B)" %
			(model_name, model.model_const_ptr, model.model_const_size, model_name,
			 ref_const_base, len(const_bytes)))

	ref_cmd_base, cmd_buffer_bytes = syms.by_name("cmd_buffer_%s" % model_name)
	if model.cmd_buffer_ptr != ref_cmd_base or model.cmd_buffer_len * 4 != len(cmd_buffer_bytes):
		raise ValueError(
			"model_%s.cmd_buffer_ptr/cmd_buffer_len (0x%08x, %u words) do not match "
			"symbol cmd_buffer_%s (0x%08x, %u B)" %
			(model_name, model.cmd_buffer_ptr, model.cmd_buffer_len, model_name,
			 ref_cmd_base, len(cmd_buffer_bytes)))
	if len(cmd_buffer_bytes) % 4 != 0:
		raise ValueError("cmd_buffer_%s size (%u B) is not a multiple of 4" %
				  (model_name, len(cmd_buffer_bytes)))

	extra_outputs_bytes = b""
	if model.extra_output_cnt > 0:
		extra_sym_name, extra_outputs_bytes = syms.by_address(model.extra_outputs)
		expected_len = model.extra_output_cnt * ctypes.sizeof(OutputDesc)
		if len(extra_outputs_bytes) != expected_len:
			raise ValueError(
				"model_%s.extra_outputs (symbol '%s', %u B) does not match "
				"extra_output_cnt=%u * sizeof(nrf_axon_compiled_model_output_s) "
				"= %u B" % (model_name, extra_sym_name, len(extra_outputs_bytes),
					    model.extra_output_cnt, expected_len))

	# nrf_axon_interlayer_buffer is app-owned RAM: its address is only known to whatever
	# firmware actually links this package's loader, not to this reference build, so
	# pointer fields that target it are stored as an *offset* rather than an address.
	interlayer_addr, _ = syms.by_name("nrf_axon_interlayer_buffer")
	relocate_ram_pointers(model, model_name, interlayer_addr)

	# Everything below is flash-owned model data: the struct itself, cmd_buffer, model_const,
	# and (optionally) extra_outputs are laid out back-to-back starting right after the
	# header, at addresses fully determined by --address, exactly as cmd_buffer/model_const
	# already were in prior format versions - so their pointers can be relocated here on the
	# host, once, instead of by the on-device loader at every boot.
	struct_len = len(buf)
	struct_base = address + HEADER_LEN
	cmd_buffer_base = struct_base + struct_len
	model_const_base = cmd_buffer_base + len(cmd_buffer_bytes)
	extra_outputs_base = model_const_base + len(const_bytes)

	cmd_buffer_bytes = relocate_cmd_buffer(cmd_buffer_bytes, ref_const_base, model_const_base,
						len(const_bytes))
	relocate_flash_pointers(model, cmd_buffer_base, model_const_base, extra_outputs_base)

	section_len = [struct_len, len(cmd_buffer_bytes), len(const_bytes), len(extra_outputs_bytes)]
	payload = bytes(buf) + cmd_buffer_bytes + const_bytes + extra_outputs_bytes
	payload_size = len(payload)

	name = model_name.encode("utf-8")[:NAME_LEN]
	name = name + b"\x00" * (NAME_LEN - len(name))

	def pack(crc):
		return struct.pack(
			HEADER_FMT,
			MAGIC, FORMAT_VERSION, MODEL_TYPE_AXON, 0, name, version, payload_size,
			*section_len,
			struct_len, struct_base,
			crc,
		)

	header_no_crc = pack(0)
	assert len(header_no_crc) == HEADER_LEN, \
		"HEADER_FMT (%u B) must match struct model_pkg_axon_header (see model_pkg.h)" % HEADER_LEN
	crc = zlib.crc32(header_no_crc + payload) & 0xFFFFFFFF

	print("package_base=0x%08x struct=%u B cmd_buffer=%u B model_const=%u B extra_outputs=%u B" %
	      (struct_base, section_len[0], section_len[1], section_len[2], section_len[3]))

	return pack(crc) + payload


def write_intel_hex(path, data, base_address):
	"""Minimal Intel HEX writer (data + extended linear address + EOF records)."""
	lines = []
	ext_high = -1
	chunk_size = 16

	for offset in range(0, len(data), chunk_size):
		address = base_address + offset
		high = (address >> 16) & 0xFFFF
		if high != ext_high:
			rec = bytes([2, 0, 0, 0x04, high >> 8, high & 0xFF])
			checksum = (-sum(rec)) & 0xFF
			lines.append(":" + rec.hex().upper() + "%02X" % checksum)
			ext_high = high

		chunk = data[offset:offset + chunk_size]
		low = address & 0xFFFF
		rec = bytes([len(chunk), low >> 8, low & 0xFF, 0x00]) + chunk
		checksum = (-sum(rec)) & 0xFF
		lines.append(":" + rec.hex().upper() + "%02X" % checksum)

	lines.append(":00000001FF")
	path.write_text("\n".join(lines) + "\n")


def main():
	parser = argparse.ArgumentParser(description=__doc__,
					  formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--elf", type=Path, required=True,
			     help="Path to a reference build's zephyr.elf "
				  "(*_REFERENCE_BUILD=y, never flashed)")
	parser.add_argument("--model-name", default="hello_axon",
			     help="Model name, matching the *_<name> suffix used by the "
				  "generated header's symbols (default: hello_axon)")
	parser.add_argument("--version", default="1.0.0",
			     help="Package version, e.g. 1.0.0 (default: 1.0.0)")
	parser.add_argument("-o", "--out", type=Path, default=None,
			     help="Output basename (default: --model-name)")
	parser.add_argument("--dts", type=Path, default=None,
			     help="Path to a build's generated zephyr.dts (e.g. "
				  "build_ref/zephyr/zephyr.dts) to read the model_storage "
				  "partition's actual address and size from, instead of trusting "
				  "--address/--partition-size to still match it")
	parser.add_argument("--address", type=lambda x: int(x, 0), default=None,
			     help="Absolute flash address of the model_storage partition "
				  "(default: 0x102000, nRF54LM20 DK 'model_storage' partition, "
				  "unless --dts is given). Must exactly match the target board's "
				  "model_partition overlay: it is baked into every flash-owned "
				  "pointer field, not just used for the .hex file's addressing, "
				  "and the on-device loader rejects the package if it doesn't "
				  "match.")
	parser.add_argument("--partition-size", type=lambda x: int(x, 0), default=None,
			     help="Size in bytes of the model_storage partition, used only to "
				  "preflight-check the package fits (default: 968 KiB, the "
				  "nRF54LM20 DK 'model_storage' partition, unless --dts is given)")
	parser.add_argument("--no-hex", action="store_true",
			     help="Only write the raw .bin package, skip the .hex file")
	args = parser.parse_args()

	if args.dts is not None:
		dts_address, dts_partition_size = read_partition_layout_from_dts(args.dts)
	else:
		dts_address, dts_partition_size = None, None

	address = args.address if args.address is not None else \
		(dts_address if dts_address is not None else DEFAULT_ADDRESS)
	partition_size = args.partition_size if args.partition_size is not None else \
		(dts_partition_size if dts_partition_size is not None else DEFAULT_PARTITION_SIZE)

	package = build_package(args.elf, args.model_name, parse_version(args.version), address)
	check_package_fits(len(package), partition_size, str(args.elf))
	report_package_usage(len(package), partition_size)

	out_base = args.out or Path(args.model_name)
	bin_path = out_base.with_suffix(".bin")
	bin_path.write_bytes(package)
	print("Wrote %s (%u bytes)" % (bin_path, len(package)))

	if not args.no_hex:
		hex_path = out_base.with_suffix(".hex")
		write_intel_hex(hex_path, package, address)
		print("Wrote %s (base address 0x%08x)" % (hex_path, address))
		print("Flash with e.g.:")
		print("  nrfutil device program --firmware %s \\" % hex_path)
		print("      --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM")


if __name__ == "__main__":
	sys.exit(main())
