#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side packaging tool for the model-only OTA update PoC (Neuton, f32 precision).

Reads a plain-JSON model description (see models/*.json for examples) and produces a
"model package": a small header (magic/version/CRC32/section lengths, matching
include/model_ota/model_pkg.h) followed by the model's raw arrays concatenated in a fixed
order. No addresses are embedded anywhere in a Neuton package, so - unlike the planned Axon
follow-up - this tool does not need to know anything about how the firmware was linked.

Usage:
    python3 package_model.py models/regression_v1.json -o model_v1 --address 0x102000

Produces model_v1.bin (raw package) and model_v1.hex (same bytes, addressed for flashing
directly into the model_storage partition, independently of the application image), e.g.:

    nrfutil device program --firmware model_v1.hex \\
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

--address defaults to 0x102000, the offset of the dedicated "model_storage" partition
(model_partition) on the nRF54LM20 DK as of this PoC. Pass --dts pointing at a build's
generated zephyr.dts instead to read the partition's actual address and size (used to
preflight-check the package fits) straight from it, rather than trusting a hand-typed
--address/--partition-size to still match your board's overlay.
"""
import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

from model_partition_layout import (
	DEFAULT_ADDRESS,
	DEFAULT_PARTITION_SIZE,
	check_package_fits,
	read_partition_layout_from_dts,
	report_package_usage,
)

MAGIC = b"NEAI"
FORMAT_VERSION = 3
MODEL_TYPE_NEUTON = 0
NAME_LEN = 16

# Fixed section order; must match enum model_pkg_neuton_section in
# include/model_ota/model_pkg.h. Grouped by element size (float, then uint16_t, then the
# single uint8_t section last) so every section boundary lands on a naturally aligned address
# for the next section's element type - see the comment on that enum for why this matters.
SECTIONS = [
	("weights", "f"),
	("act_weights", "f"),
	("output_scale_min", "f"),
	("output_scale_max", "f"),
	("neuron_links", "H"),
	("neuron_internal_links_num", "H"),
	("neuron_external_links_num", "H"),
	("output_neurons_indices", "H"),
	("neuron_act_type_mask", "B"),
]

# <magic(4s) format_version(H) model_type(B) reserved0(B) name(16s) model_version(I)
#  payload_size(I) section_len[9](9I) crc32(I)  -- packed, little-endian, no padding.
HEADER_FMT = "<4sHBB16sII9II"
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


def build_package(model):
	name = model["name"].encode("utf-8")[:NAME_LEN]
	name = name + b"\x00" * (NAME_LEN - len(name))
	version = parse_version(model["version"])

	section_bytes = []
	section_len = []
	for key, elem_fmt in SECTIONS:
		values = model[key]
		packed = struct.pack("<%u%s" % (len(values), elem_fmt), *values)
		section_bytes.append(packed)
		section_len.append(len(packed))

	payload = b"".join(section_bytes)
	payload_size = len(payload)

	header_no_crc = struct.pack(
		HEADER_FMT,
		MAGIC,
		FORMAT_VERSION,
		MODEL_TYPE_NEUTON,
		0,
		name,
		version,
		payload_size,
		*section_len,
		0,  # crc32 placeholder, zeroed for the CRC computation itself
	)
	crc = zlib.crc32(header_no_crc + payload) & 0xFFFFFFFF

	header = struct.pack(
		HEADER_FMT,
		MAGIC,
		FORMAT_VERSION,
		MODEL_TYPE_NEUTON,
		0,
		name,
		version,
		payload_size,
		*section_len,
		crc,
	)
	assert len(header) == HEADER_LEN
	return header + payload


def sanity_check(model):
	neurons_num = len(model["neuron_internal_links_num"])
	if len(model["neuron_external_links_num"]) != neurons_num:
		raise ValueError("neuron_internal_links_num and neuron_external_links_num must have "
				  "the same length (neurons_num)")

	outputs_num = len(model["output_neurons_indices"])
	if len(model["output_scale_min"]) != outputs_num or \
	   len(model["output_scale_max"]) != outputs_num:
		raise ValueError("output_neurons_indices, output_scale_min and output_scale_max "
				  "must all have the same length (outputs_num)")

	weights_num = len(model["weights"])
	if len(model["neuron_links"]) != weights_num:
		raise ValueError("weights and neuron_links must have the same length "
				  "(weights_num)")


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
	parser.add_argument("model_json", type=Path, help="Path to a model JSON description")
	parser.add_argument("-o", "--out", type=Path, default=None,
			     help="Output basename (default: same as model_json, no extension)")
	parser.add_argument("--dts", type=Path, default=None,
			     help="Path to a build's generated zephyr.dts (e.g. "
				  "build/zephyr/zephyr.dts) to read the model_storage "
				  "partition's actual address and size from, instead of trusting "
				  "--address/--partition-size to still match it")
	parser.add_argument("--address", type=lambda x: int(x, 0), default=None,
			     help="Absolute flash address of the model_storage partition "
				  "(default: 0x102000, nRF54LM20 DK 'model_storage' partition, "
				  "unless --dts is given)")
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

	model = json.loads(args.model_json.read_text())
	sanity_check(model)
	package = build_package(model)
	check_package_fits(len(package), partition_size, str(args.model_json))
	report_package_usage(len(package), partition_size)

	out_base = args.out or args.model_json.with_suffix("")
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
