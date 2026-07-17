#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Host-side packaging tool for the model-only OTA update PoC (Neuton, f32/q16/q8 precision).

Unlike package_model.py, this tool needs no hand-written JSON: it regex-parses a Neuton
codegen's generated model source directly (e.g. nrf_edgeai_generated/Neuton/
nrf_edgeai_user_model.c under any of the Neuton samples in this repo) and builds a package
straight from it. No addresses are embedded anywhere in a Neuton package (unlike Axon), so the
generated .c file's #define counts and static const arrays are all that is needed - no reference
build or ELF introspection required.

Usage:
    python3 package_model_neuton.py \\
        ../../samples/nrf_edgeai/regression/src/nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c \\
        --name aq_regression --version 1.0.0 -o model_v1 --address 0x102000

Produces model_v1.bin (raw package) and model_v1.hex (same bytes, addressed for flashing
directly into the model_storage partition, independently of the application image), e.g.:

    nrfutil device program --firmware model_v1.hex \\
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

--address defaults to 0x102000, the offset of the dedicated "model_storage" partition
(model_partition) on the nRF54LM20 DK as of this PoC. Pass --dts pointing at a build's generated
zephyr.dts instead to read the partition's actual address and size straight from it.

Known limitations (inherited from the package format, not specific to this tool):
- MODEL_PARAMS_TYPE must be f32, q16, or q8 (the only precisions the generated codegen and the
  package format define); the resulting package records which one so the on-device loader
  knows which nrf_edgeai_model_neuton_params_u member (and neurons_buf element type) to use.
- Regression/anomaly-detection models carry their MODEL_OUTPUT_SCALE_MIN/MAX in the package
  (needed to decode their output back into physical units); classification models have no
  such arrays to begin with (nrf_edgeai_decoded_output_classif_t has no scale concept) and are
  packaged with empty output-scale sections instead.
- Anomaly-detection models additionally carry MODEL_AVERAGE_EMBEDDING (needed by
  nrf_edgeai_decoded_output_anomaly_t); regression/classification models have no such array
  and are packaged with an empty average-embedding section instead.
package_model.py's hand-written JSON path remains available for synthetic/hand-edited variants
that have no corresponding generated .c file (see
samples/nrf_edgeai/regression/src/nrf_edgeai_generated/Neuton/regression_v2.json).
"""
import argparse
import re
import sys
from pathlib import Path

from model_partition_layout import (
	DEFAULT_ADDRESS,
	DEFAULT_PARTITION_SIZE,
	check_package_fits,
	read_partition_layout_from_dts,
	report_package_usage,
)
from package_model import build_package, sanity_check, write_intel_hex

# Neuton task codes (see __NRF_EDGEAI_TASK_* in include/nrf_edgeai/rt/nrf_edgeai_model_types.h).
TASK_MULT_CLASS = 0
TASK_BIN_CLASS = 1
TASK_REGRESSION = 2
TASK_ANOMALY_DETECTION = 3

# Tasks whose generated source emits MODEL_OUTPUT_SCALE_MIN/MAX / MODEL_AVERAGE_EMBEDDING.
# Classification tasks emit neither (nrf_edgeai_decoded_output_classif_t has no scale concept);
# regression additionally has no average embedding (nrf_edgeai_decoded_output_regress_t has no
# such field) - only anomaly detection uses both. Missing arrays not listed here are packaged
# empty instead of raising, keyed by the JSON key arrays_for() maps them to (see OPTIONAL_ARRAY_TASKS).
TASKS_WITH_OUTPUT_SCALE = (TASK_REGRESSION, TASK_ANOMALY_DETECTION)
TASKS_WITH_AVERAGE_EMBEDDING = (TASK_ANOMALY_DETECTION,)
OPTIONAL_ARRAY_TASKS = {
	"output_scale_min": TASKS_WITH_OUTPUT_SCALE,
	"output_scale_max": TASKS_WITH_OUTPUT_SCALE,
	"average_embedding": TASKS_WITH_AVERAGE_EMBEDDING,
}

# MODEL_PARAMS_TYPE string (as it appears verbatim in the generated source) -> model["params_type"]
# value; must match enum model_pkg_neuton_params_type in include/model_ota/model_pkg.h.
PARAMS_TYPES = {"f32": 0, "q16": 1, "q8": 2}

# Maps a generated .c array name to the JSON key build_package()/sanity_check() expect, and
# whether that array's C literals should be parsed as float or int. weights/act_weights are
# only float for an f32 model - q16/q8 ones store (quantized) integers instead.
def arrays_for(params_type):
	weight_elem_type = float if params_type == PARAMS_TYPES["f32"] else int
	return [
		("MODEL_WEIGHTS", "weights", weight_elem_type),
		("MODEL_NEURON_ACTIVATION_WEIGHTS", "act_weights", weight_elem_type),
		("MODEL_OUTPUT_SCALE_MIN", "output_scale_min", float),
		("MODEL_OUTPUT_SCALE_MAX", "output_scale_max", float),
		("MODEL_AVERAGE_EMBEDDING", "average_embedding", float),
		("MODEL_NEURONS_LINKS", "neuron_links", int),
		("MODEL_NEURON_INTERNAL_LINKS_NUM", "neuron_internal_links_num", int),
		("MODEL_NEURON_EXTERNAL_LINKS_NUM", "neuron_external_links_num", int),
		("MODEL_OUTPUT_NEURONS_INDICES", "output_neurons_indices", int),
		("MODEL_NEURON_ACTIVATION_TYPE_MASK", "neuron_act_type_mask", int),
	]


def extract_define(source, name):
	match = re.search(r"#define\s+%s\s+(\S+)" % re.escape(name), source)
	if not match:
		raise ValueError("could not find '#define %s' in the generated model source" % name)
	return match.group(1)


def extract_array(source, array_name):
	"""Returns the raw comma-separated literal text inside `static const ... array_name[] =
	{ ... };`, or None if the array does not appear in the source at all (expected for the
	OPTIONAL_ARRAYS on classification models)."""
	match = re.search(
		r"static\s+const\s+\S+\s+%s\s*\[\s*\]\s*=\s*\{(.*?)\}\s*;" % re.escape(array_name),
		source, re.DOTALL)
	if not match:
		return None
	body = match.group(1).strip()
	if not body:
		return []
	# Literals may be split across lines with arbitrary whitespace; a trailing comma before
	# the closing brace is not used by the Neuton codegen, but tolerate it anyway.
	return [item.strip() for item in body.split(",") if item.strip()]


def parse_neuton_model_c(source, name, version):
	params_type_str = extract_define(source, "MODEL_PARAMS_TYPE")
	if params_type_str not in PARAMS_TYPES:
		raise ValueError(
			"MODEL_PARAMS_TYPE is '%s', but package_model_neuton.py only supports "
			"%s" % (params_type_str, ", ".join(sorted(PARAMS_TYPES))))
	params_type = PARAMS_TYPES[params_type_str]

	task = int(extract_define(source, "MODEL_TASK"))

	model = {"name": name, "version": version, "params_type": params_type}
	for array_name, json_key, elem_type in arrays_for(params_type):
		literals = extract_array(source, array_name)
		if literals is None:
			required_for_tasks = OPTIONAL_ARRAY_TASKS.get(json_key)
			if required_for_tasks is not None and task not in required_for_tasks:
				# This task's generated source never defines this array (see
				# OPTIONAL_ARRAY_TASKS) - pack an empty section instead of failing.
				model[json_key] = []
				continue
			if required_for_tasks is not None:
				raise ValueError(
					"MODEL_TASK=%u should define %s, but it was not found in "
					"the generated model source" % (task, array_name))
			raise ValueError(
				"could not find 'static const ... %s[]' in the generated model "
				"source" % array_name)
		# int(v, 0) auto-detects the base, so both "0x1" and plain decimal literals work.
		model[json_key] = [int(v, 0) if elem_type is int else float(v) for v in literals]

	return model


def main():
	parser = argparse.ArgumentParser(description=__doc__,
					  formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("model_c", type=Path,
			     help="Path to a Neuton codegen's generated nrf_edgeai_user_model.c")
	parser.add_argument("--name", required=True,
			     help="Model name to embed in the package header (<=16 bytes, "
				  "truncated if longer); the generated source carries no name "
				  "of its own")
	parser.add_argument("--version", required=True,
			     help="Package version, e.g. 1.0.0")
	parser.add_argument("-o", "--out", type=Path, default=None,
			     help="Output basename (default: same as model_c, no extension)")
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

	source = args.model_c.read_text()
	model = parse_neuton_model_c(source, args.name, args.version)
	sanity_check(model)
	package = build_package(model)
	check_package_fits(len(package), partition_size, str(args.model_c))
	report_package_usage(len(package), partition_size)

	out_base = args.out or args.model_c.with_suffix("")
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
