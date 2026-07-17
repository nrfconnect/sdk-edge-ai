#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side packaging tool for the model-only OTA update PoC (Neuton, f32/q16/q8 precision).

Reads a plain-JSON model description (see samples/nrf_edgeai/regression/src/
nrf_edgeai_generated/Neuton/regression_v2.json for an example) and produces a
"model package": a small header (magic/version/CRC32/section lengths, matching
include/model_ota/model_pkg.h) followed by the model's raw arrays concatenated in a fixed
order. No addresses are embedded anywhere in a Neuton package, so - unlike Axon (see
package_model_axon.py) - this tool does not need to know anything about how the firmware was
linked. Meant for models with no corresponding generated .c file (hand-authored/synthetic ones,
or a model trained by some other pipeline entirely) - if one exists, use
package_model_neuton.py against it directly instead, no hand-transcription into JSON needed.

Usage:
    python3 package_model.py \\
        ../../samples/nrf_edgeai/regression/src/nrf_edgeai_generated/Neuton/regression_v2.json \\
        -o model_v2 --address 0x102000

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
FORMAT_VERSION = 9  # must match MODEL_PKG_FORMAT_VERSION in include/model_ota/model_pkg.h
MODEL_TYPE_NEUTON = 0
NAME_LEN = 16

# model["params_type"] value -> struct.pack() element format for the weights/act_weights
# sections; must match enum model_pkg_neuton_params_type in include/model_ota/model_pkg.h
# and nrf_edgeai_model_neuton_params_{f32,q16,q8}_t (weights signed, act_weights unsigned).
PARAMS_TYPE_WEIGHTS_FMT = {0: "f", 1: "h", 2: "b"}
PARAMS_TYPE_ACT_WEIGHTS_FMT = {0: "f", 1: "H", 2: "B"}


def section_specs(params_type):
	"""Section order and struct.pack() element format, keyed by model dict key; must match
	enum model_pkg_neuton_section in include/model_ota/model_pkg.h. weights/act_weights vary
	with params_type (see PARAMS_TYPE_*_FMT above); every other section is precision-
	independent."""
	return [
		("weights", PARAMS_TYPE_WEIGHTS_FMT[params_type]),
		("act_weights", PARAMS_TYPE_ACT_WEIGHTS_FMT[params_type]),
		("output_scale_min", "f"),
		("output_scale_max", "f"),
		("average_embedding", "f"),
		("neuron_links", "H"),
		("neuron_internal_links_num", "H"),
		("neuron_external_links_num", "H"),
		("output_neurons_indices", "H"),
		("neuron_act_type_mask", "B"),
	]


# <magic(4s) format_version(H) model_type(B) params_type(B) name(16s) model_version(I)
#  payload_size(I) section_len[10](10I) crc32(I)  -- packed, little-endian, no padding.
HEADER_FMT = "<4sHBB16sII10II"
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
	params_type = model.get("params_type", 0)

	# Each section is placed at the next 4-byte-aligned offset from the payload's own start
	# (itself 4-byte aligned, see include/model_ota/model_pkg.h) - pad with zero bytes as
	# needed so q16/q8 weights/act_weights never leave what follows them misaligned. This
	# padding is not reflected in section_len (kept as the exact array length, so the loader
	# can still derive element counts from it), only in the payload's total byte count.
	section_bytes = []
	section_len = []
	offset = 0
	for key, elem_fmt in section_specs(params_type):
		pad = -offset % 4
		if pad:
			section_bytes.append(b"\x00" * pad)
			offset += pad
		values = model[key]
		packed = struct.pack("<%u%s" % (len(values), elem_fmt), *values)
		section_bytes.append(packed)
		section_len.append(len(packed))
		offset += len(packed)

	payload = b"".join(section_bytes)
	payload_size = len(payload)

	header_no_crc = struct.pack(
		HEADER_FMT,
		MAGIC,
		FORMAT_VERSION,
		MODEL_TYPE_NEUTON,
		params_type,
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
		params_type,
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
	# Classification models have no output-scale concept at all (see
	# nrf_edgeai_decoded_output_classif_t) and are packaged with output_scale_min/max both
	# empty regardless of outputs_num - only enforce the length match when scale is used.
	if model["output_scale_min"] or model["output_scale_max"]:
		if len(model["output_scale_min"]) != outputs_num or \
		   len(model["output_scale_max"]) != outputs_num:
			raise ValueError("output_neurons_indices, output_scale_min and "
					  "output_scale_max must all have the same length "
					  "(outputs_num)")

	# average_embedding (nrf_edgeai_decoded_output_anomaly_t) is only meaningful for anomaly
	# detection models and is packaged empty otherwise - same rule as output_scale above.
	if model["average_embedding"] and len(model["average_embedding"]) != outputs_num:
		raise ValueError("average_embedding must have the same length as "
				  "output_neurons_indices (outputs_num)")

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
