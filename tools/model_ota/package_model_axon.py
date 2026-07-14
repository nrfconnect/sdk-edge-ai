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
fields.

One RAM-owned reference is not covered by that struct-field relocation: cmd_buffer's own
embedded literal references to nrf_axon_interlayer_buffer (used for inter-layer data handoff
between the NPU's compiled instructions), which - unlike its references into model_const - are
never relocated, by this tool or the on-device loader. They are only ever correct if this
reference build's nrf_axon_interlayer_buffer address happens to match the deployed device's.
This tool cannot fix that at packaging time (the deployed address isn't known yet), so it
instead records the reference build's address in the package (interlayer_addr) purely so the
loader can detect a mismatch and refuse to load rather than silently mispredict - see
model_pkg_load_axon() for the actual check. Packaging therefore needs two inputs:

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

Classification models using `labels` (per-class text labels) are supported: the labels array
and every individual label string it points to are extracted straight from the reference
build's ELF (see ElfSymbols.read_cstring()), repacked into their own package sections, and
relocated exactly like model_const - no runtime patching needed, no separate class count field
either (it is derived from output_dimensions, see num_output_classes()).

Known limitations (see the plan this tool was implemented from for the reasoning): models using
`persistent_vars` (streaming/VarHandle models), or CPU op extensions (whose addresses cmd_buffer
embeds literally, exactly like nrf_axon_interlayer_buffer, but with no runtime mismatch check
backing them up - see check_no_op_extension_refs()) are rejected outright. Computing
per-persistent-variable buffer offsets, and relocating op extension references at load time the
way nrf_axon_interlayer_buffer's offset-based scheme does, were judged not worth the added
complexity for a PoC with no such model to validate against yet. Multi-input models (input_cnt >
1, both external) and models with extra_outputs are supported by this tool and the loader, but -
absent such a model in this repo - are only exercised by this tool's own unit tests, not on real
hardware.
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
FORMAT_VERSION = 5
MODEL_TYPE_AXON = 1
NAME_LEN = 16

# <magic(4s) format_version(H) model_type(B) reserved0(B) name(16s) model_version(I)
#  payload_size(I) section_len[6](6I) struct_size(I) package_base(I) interlayer_addr(I)
#  crc32(I) -- packed, little-endian, no padding. section_len is
#  [struct, cmd_buffer, model_const, extra_outputs, labels, label_strings].
HEADER_FMT = "<4sHBB16sII6III" "II"
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
	"""Thin wrapper around an ELF's .symtab for the lookups this tool needs: by name (for
	symbols whose generated name we know, e.g. model_<name>), by address (for pointer fields
	whose target's name isn't predictable, e.g. extra_outputs), and raw address-range reads
	(for label strings, which typically have no symbol of their own - see read_cstring())."""

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
			# Every loaded section's address range, regardless of whether any symbol
			# covers it - string literals are commonly merged into anonymous,
			# mergeable .rodata.str sections with no symbol table entry at all.
			self._all_sections = []
			for section in elffile.iter_sections():
				if section["sh_addr"] != 0 and section["sh_type"] != "SHT_NOBITS":
					self._all_sections.append(
						(section["sh_addr"], section.data(), section.name))

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

	def read_cstring(self, address, max_len=4096):
		"""Returns the NUL-terminated byte string (excluding the NUL) stored at `address` -
		e.g. one of labels[]'s per-class strings. Looks it up by raw section address
		range rather than by symbol (unlike by_name()/by_address()), since compilers
		commonly merge string literals into anonymous, mergeable .rodata.str1.1-style
		sections with no symbol table entry of their own."""
		for sh_addr, data, section_name in self._all_sections:
			if sh_addr <= address < sh_addr + len(data):
				offset = address - sh_addr
				nul = data.find(b"\x00", offset, offset + max_len)
				if nul == -1:
					raise ValueError(
						"No NUL terminator found within %u bytes of address "
						"0x%08x in section '%s' of %s" %
						(max_len, address, section_name, self.elf_path))
				return data[offset:nul]
		raise ValueError("No section in %s covers address 0x%08x" %
				  (self.elf_path, address))

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


def check_no_op_extension_refs(cmd_buffer_bytes, model_name, op_extension_addrs):
	"""Raises ValueError if cmd_buffer's words reference any of op_extension_addrs (a CPU op
	extension function's address, e.g. nrf_axon_nn_op_extension_relu). Those addresses are
	baked in by the reference build's own link and - unlike cmd_buffer's references into
	model_const - are never relocated, by this tool or the on-device loader, so they are only
	ever correct if the reference build's .text layout happens to match the deployed app's
	exactly. Pure function - no ELF access - so it can be unit-tested directly."""
	if not op_extension_addrs:
		return
	words = struct.unpack("<%uI" % (len(cmd_buffer_bytes) // 4), cmd_buffer_bytes)
	word_set = set(words)
	for addr in op_extension_addrs:
		# Thumb code pointers have bit 0 set in the stored address.
		if addr in word_set or (addr | 1) in word_set:
			raise ValueError(
				"model_%s's cmd_buffer references a CPU op extension at 0x%08x - "
				"packaging models that use op extensions is not supported yet: "
				"their addresses are baked in by this reference build's own link "
				"and are never relocated, so they would only be correct on a "
				"deployed device whose .text layout happens to match this "
				"reference build's exactly" % (model_name, addr))


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


def relocate_flash_pointers(model, cmd_buffer_base, model_const_base, extra_outputs_base,
			     labels_base=0):
	"""Rewrites model's flash-owned pointer fields to their final address once flashed into
	model_storage. Pure function - no ELF access - so it can be unit-tested directly against
	synthetic CompiledModel instances built from raw bytes."""
	model.cmd_buffer_ptr = cmd_buffer_base
	model.model_const_ptr = model_const_base
	if model.extra_output_cnt > 0:
		model.extra_outputs = extra_outputs_base
	if model.labels != 0:
		model.labels = labels_base


def num_output_classes(model):
	"""Returns the number of classification classes/labels a model's output_dimensions
	implies - the same element count findmax{8,16,32}() (see nrf_axon_nn_infer.c) scans over
	to pick the highest-scoring index, which labels[] is then indexed by. Pure function - no
	ELF access - so it can be unit-tested directly."""
	dim = model.output_dimensions
	return dim.height * dim.width * dim.channel_cnt


def build_labels_section(label_strings, strings_base):
	"""Packs `label_strings` (a list of raw bytes, one per classification label, in index
	order - i.e. label_strings[i] is what labels[i] should point to) into a single
	NUL-separated blob plus a matching array of pointers into it, assuming that blob will be
	placed at `strings_base` once flashed. Returns (labels_bytes, label_strings_bytes). Pure
	function - no ELF access - so it can be unit-tested directly."""
	offsets = []
	blob_parts = []
	running_offset = 0
	for s in label_strings:
		if b"\x00" in s:
			raise ValueError("label %r contains an embedded NUL byte" % s)
		offsets.append(running_offset)
		blob_parts.append(s + b"\x00")
		running_offset += len(s) + 1
	label_strings_bytes = b"".join(blob_parts)
	labels_bytes = struct.pack("<%uI" % len(label_strings),
				    *(strings_base + o for o in offsets))
	return labels_bytes, label_strings_bytes


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

	op_extension_addrs = [sym["st_value"] & ~1 for sym in syms.symbols
			       if sym.name.startswith("nrf_axon_nn_op_extension_")]
	check_no_op_extension_refs(cmd_buffer_bytes, model_name, op_extension_addrs)

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

	labels_bytes = b""
	label_strings_bytes = b""
	if model.labels != 0:
		num_labels = num_output_classes(model)
		if num_labels == 0:
			raise ValueError(
				"model_%s has a non-NULL 'labels' pointer but an empty "
				"output_dimensions (0 classes)" % model_name)
		_, labels_array_bytes = syms.by_address(model.labels)
		if len(labels_array_bytes) != num_labels * 4:
			raise ValueError(
				"model_%s.labels array (%u B) does not match output_dimensions' "
				"implied class count=%u (%u B expected)" %
				(model_name, len(labels_array_bytes), num_labels, num_labels * 4))
		label_ptrs = struct.unpack("<%uI" % num_labels, labels_array_bytes)
		label_strings = [syms.read_cstring(ptr) for ptr in label_ptrs]

	# nrf_axon_interlayer_buffer is app-owned RAM: its address is only known to whatever
	# firmware actually links this package's loader, not to this reference build, so
	# pointer fields that target it are stored as an *offset* rather than an address.
	interlayer_addr, _ = syms.by_name("nrf_axon_interlayer_buffer")
	relocate_ram_pointers(model, model_name, interlayer_addr)

	# Everything below is flash-owned model data: the struct itself, cmd_buffer, model_const,
	# and (optionally) extra_outputs/labels are laid out back-to-back starting right after
	# the header, at addresses fully determined by --address, exactly as cmd_buffer/
	# model_const already were in prior format versions - so their pointers can be
	# relocated here on the host, once, instead of by the on-device loader at every boot.
	struct_len = len(buf)
	struct_base = address + HEADER_LEN
	cmd_buffer_base = struct_base + struct_len
	model_const_base = cmd_buffer_base + len(cmd_buffer_bytes)
	extra_outputs_base = model_const_base + len(const_bytes)
	labels_base = extra_outputs_base + len(extra_outputs_bytes)

	if model.labels != 0:
		label_strings_base = labels_base + num_labels * 4
		labels_bytes, label_strings_bytes = build_labels_section(label_strings,
									   label_strings_base)

	cmd_buffer_bytes = relocate_cmd_buffer(cmd_buffer_bytes, ref_const_base, model_const_base,
						len(const_bytes))
	relocate_flash_pointers(model, cmd_buffer_base, model_const_base, extra_outputs_base,
				 labels_base)

	section_len = [struct_len, len(cmd_buffer_bytes), len(const_bytes), len(extra_outputs_bytes),
		       len(labels_bytes), len(label_strings_bytes)]
	payload = (bytes(buf) + cmd_buffer_bytes + const_bytes + extra_outputs_bytes +
		   labels_bytes + label_strings_bytes)
	payload_size = len(payload)

	name = model_name.encode("utf-8")[:NAME_LEN]
	name = name + b"\x00" * (NAME_LEN - len(name))

	def pack(crc):
		return struct.pack(
			HEADER_FMT,
			MAGIC, FORMAT_VERSION, MODEL_TYPE_AXON, 0, name, version, payload_size,
			*section_len,
			struct_len, struct_base, interlayer_addr,
			crc,
		)

	header_no_crc = pack(0)
	assert len(header_no_crc) == HEADER_LEN, \
		"HEADER_FMT (%u B) must match struct model_pkg_axon_header (see model_pkg.h)" % HEADER_LEN
	crc = zlib.crc32(header_no_crc + payload) & 0xFFFFFFFF

	print("package_base=0x%08x struct=%u B cmd_buffer=%u B model_const=%u B extra_outputs=%u B "
	      "labels=%u B (%u B strings)" %
	      (struct_base, section_len[0], section_len[1], section_len[2], section_len[3],
	       section_len[4], section_len[5]))

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
