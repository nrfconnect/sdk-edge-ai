#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side packaging tool for the model-only OTA update PoC (Axon).

This tool turns a "model stub" ELF - a tiny, standalone link of just a generated Axon model
header, produced by lib/model_ota/cmake/nrf_axon_model_stub.cmake as part of a normal `west
build` (see doc/libraries/model_ota.rst for the full picture) - into a flashable model package.

The model stub is linked with its `.model_stub` output section placed exactly at the
model_storage partition's own address plus this tool's header size (see MODEL_STUB_ADDR in
nrf_axon_model_stub.cmake, and model_stub_axon.ld), and with every app-owned symbol it
references (nrf_axon_interlayer_buffer, nrf_axon_nn_op_extension_*,
axon_model_<name>_persistent_vars) resolved to the real address the *deployed application*
placed them at (gen_axon_stub_fixups.py + extract_elf_syms.py). That means every pointer field
inside the compiler-generated nrf_axon_nn_compiled_model_s struct - flash-owned (cmd_buffer,
model_const, extra_outputs, labels) or app-owned RAM alike - is already the final, correct
absolute address for this specific application, by construction of the link itself. This tool
therefore does not relocate anything: it copies the model stub's `.model_stub` section verbatim
as the package payload, and just records where the model struct starts within it
(struct_offset) plus a package_base the on-device loader can cross-check against.

Usage:
    python3 package_model_axon.py \\
        --elf build/hello_axon/model_stub_hello_axon/hello_axon_model_stub.elf \\
        --model-name hello_axon -o model_v1 --dts build/hello_axon/zephyr/zephyr.dts

--dts reads the model_storage partition's actual address and size straight out of a build's
generated zephyr.dts, so a stale/wrong --address can't silently produce a package that doesn't
match where the model stub was actually linked, or one that overflows the partition. Falls back
to --address (default 0x102000) and --partition-size (default 968 KiB) - the nRF54LM20 DK
values - if omitted.

Produces model_v1.bin (raw package) and model_v1.hex (same bytes, addressed for flashing
directly into the model_storage partition, independently of the application image), e.g.:

    nrfutil device program --firmware model_v1.hex \\
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

Requires pyelftools (already bundled with the NCS toolchain's Python; otherwise
`pip install pyelftools`).

Classification models using `labels` (per-class text labels), streaming/VarHandle models using
`persistent_vars`, and models using CPU op extensions are all supported: every pointer field
referring to any of them is already correct in the model stub's own link, so none of them need
special-casing here any more.

Known limitation: models with a non-NULL `packed_output_buf` (a compile-time array the
generated model header itself declares) are rejected outright - the deployed application, which
no longer links in the generated model header, has no such array to provide.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

from elftools.elf.elffile import ELFFile

from axon_struct_layout import CompiledModel, check_struct_size
from model_partition_layout import (
	DEFAULT_ADDRESS,
	DEFAULT_PARTITION_SIZE,
	check_package_fits,
	read_partition_layout_from_dts,
	report_package_usage,
)

MAGIC = b"NEAI"
FORMAT_VERSION = 6  # must match MODEL_PKG_FORMAT_VERSION in include/model_ota/model_pkg.h
MODEL_TYPE_AXON = 1
NAME_LEN = 16

# <magic(4s) format_version(H) model_type(B) reserved0(B) name(16s) model_version(I)
#  payload_size(I) struct_offset(I) struct_size(I) package_base(I) crc32(I)> -- packed,
# little-endian, no padding. Must match struct model_pkg_axon_header (see model_pkg.h).
HEADER_FMT = "<4sHBB16sIIIIII"
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
	"""Thin wrapper around a model stub ELF: looks up a named symbol's initialized data (for
	the model struct itself, e.g. model_<name>), and exposes a named section's own
	(address, raw_bytes) - used to grab the whole `.model_stub` payload blob verbatim."""

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
			self._section_cache = {}
			for sym in self.symbols:
				shndx = sym["st_shndx"]
				if not isinstance(shndx, int) or shndx in self._section_cache:
					continue
				section = elffile.get_section(shndx)
				if section is not None:
					self._section_cache[shndx] = (section["sh_addr"], section.data(),
								       section.name)
			self._named_sections = {}
			for section in elffile.iter_sections():
				self._named_sections[section.name] = (section["sh_addr"], section.data())

	def by_name(self, name):
		"""Returns (address, raw_bytes) for a named symbol's initialized data."""
		matches = [s for s in self.symbols if s.name == name]
		if not matches:
			raise ValueError(
				"Symbol '%s' not found in %s - was the model header actually linked "
				"into this model stub?" % (name, self.elf_path))
		return self._read(matches[0])

	def section(self, name):
		"""Returns (address, raw_bytes) for a named section, e.g. '.model_stub'."""
		if name not in self._named_sections:
			raise ValueError("%s has no '%s' section - was this compiled via "
					  "nrf_axon_model_stub.cmake?" % (self.elf_path, name))
		return self._named_sections[name]

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
	"""Raises ValueError for the one known-unsupported model shape (see module docstring).
	Pure function - no ELF access - so it can be unit-tested directly against synthetic
	CompiledModel instances built from raw bytes."""
	if model.packed_output_buf != 0:
		raise ValueError(
			"model_%s has a non-NULL 'packed_output_buf' - dedicated packed-output "
			"buffers are not supported (they are compile-time arrays the deployed "
			"app, which no longer links in the generated model header, cannot "
			"provide)" % model_name)
	if model.input_cnt == 0 or model.input_cnt > len(model.inputs):
		raise ValueError("model_%s has an implausible input_cnt=%u" %
				  (model_name, model.input_cnt))
	if model.extra_output_cnt > 0 and model.extra_outputs == 0:
		raise ValueError(
			"model_%s has extra_output_cnt=%u but a NULL extra_outputs pointer" %
			(model_name, model.extra_output_cnt))
	if model.extra_output_cnt == 0 and model.extra_outputs != 0:
		raise ValueError(
			"model_%s has extra_output_cnt=0 but a non-NULL extra_outputs pointer" %
			model_name)


def build_package(elf_path, model_name, version, address):
	syms = ElfSymbols(elf_path)

	payload_base, payload = syms.section(".model_stub")
	expected_payload_base = address + HEADER_LEN
	if payload_base != expected_payload_base:
		raise ValueError(
			"Model stub's .model_stub section is linked at 0x%08x, but --address "
			"0x%08x + this tool's header size (%u B) = 0x%08x - re-link the model "
			"stub with a matching MODEL_STUB_ADDR (see "
			"lib/model_ota/cmake/nrf_axon_model_stub.cmake)" %
			(payload_base, address, HEADER_LEN, expected_payload_base))

	struct_addr, struct_bytes = syms.by_name("model_%s" % model_name)
	check_struct_size(len(struct_bytes), elf_path)

	struct_offset = struct_addr - payload_base
	if struct_offset < 0 or struct_offset + len(struct_bytes) > len(payload):
		raise ValueError(
			"model_%s (0x%08x, %u B) does not fall within the .model_stub payload "
			"(0x%08x, %u B)" %
			(model_name, struct_addr, len(struct_bytes), payload_base, len(payload)))

	model = CompiledModel.from_buffer(bytearray(struct_bytes))
	validate_shape(model, model_name)

	name = model_name.encode("utf-8")[:NAME_LEN]
	name = name + b"\x00" * (NAME_LEN - len(name))
	payload_size = len(payload)

	def pack(crc):
		return struct.pack(
			HEADER_FMT,
			MAGIC, FORMAT_VERSION, MODEL_TYPE_AXON, 0, name, version, payload_size,
			struct_offset, len(struct_bytes), payload_base,
			crc,
		)

	header_no_crc = pack(0)
	assert len(header_no_crc) == HEADER_LEN, \
		"HEADER_FMT (%u B) must match struct model_pkg_axon_header (see model_pkg.h)" % HEADER_LEN
	crc = zlib.crc32(header_no_crc + payload) & 0xFFFFFFFF

	print("package_base=0x%08x payload=%u B struct_offset=%u struct_size=%u B "
	      "(cmd_buffer=%u words, model_const=%u B)" %
	      (payload_base, payload_size, struct_offset, len(struct_bytes),
	       model.cmd_buffer_len, model.model_const_size))

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
	parser.add_argument("--elf", type=Path, default=None,
			     help="Path to a model stub's ELF (see "
				  "lib/model_ota/cmake/nrf_axon_model_stub.cmake). Required "
				  "unless --print-header-len is given.")
	parser.add_argument("--model-name", default="hello_axon",
			     help="Model name, matching the *_<name> suffix used by the "
				  "generated header's symbols (default: hello_axon)")
	parser.add_argument("--version", default="1.0.0",
			     help="Package version, e.g. 1.0.0 (default: 1.0.0)")
	parser.add_argument("-o", "--out", type=Path, default=None,
			     help="Output basename (default: --model-name)")
	parser.add_argument("--dts", type=Path, default=None,
			     help="Path to a build's generated zephyr.dts (e.g. "
				  "build/hello_axon/zephyr/zephyr.dts) to read the "
				  "model_storage partition's actual address and size from, "
				  "instead of trusting --address/--partition-size to still "
				  "match it")
	parser.add_argument("--address", type=lambda x: int(x, 0), default=None,
			     help="Absolute flash address of the model_storage partition "
				  "(default: 0x102000, nRF54LM20 DK 'model_storage' partition, "
				  "unless --dts is given). Must exactly match the address the "
				  "model stub was linked at (MODEL_STUB_ADDR): the on-device "
				  "loader rejects the package if it doesn't.")
	parser.add_argument("--partition-size", type=lambda x: int(x, 0), default=None,
			     help="Size in bytes of the model_storage partition, used only to "
				  "preflight-check the package fits (default: 968 KiB, the "
				  "nRF54LM20 DK 'model_storage' partition, unless --dts is given)")
	parser.add_argument("--no-hex", action="store_true",
			     help="Only write the raw .bin package, skip the .hex file")
	parser.add_argument("--print-header-len", action="store_true",
			     help="Print HEADER_LEN (bytes) and exit - used by "
				  "nrf_axon_model_stub.cmake to size the model stub's own "
				  "header-reservation placeholder so package_base lines up "
				  "with no host-side arithmetic")
	args = parser.parse_args()

	if args.print_header_len:
		print(HEADER_LEN)
		return 0

	if args.elf is None:
		parser.error("--elf is required unless --print-header-len is given")

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
