#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Discover app-owned symbols a generated Axon model header references, and produce a patched
copy of that header safe to compile into a standalone "model stub" link (see
nrf_axon_model_stub.cmake and model_stub_axon.c).

Four kinds of symbols in a compiler-generated nrf_axon_model_<name>.h are not owned by the
model image itself - they are owned by whatever application actually deploys the model, and
their real address is only known once that application has been built:

- nrf_axon_interlayer_buffer (RAM scratch shared by every model on the device).
- nrf_axon_nn_op_extension_* (CPU fallback functions for NPU ops the model uses, e.g. softmax).
- axonpro_* driver-owned constants some cmd_buffer entries reference directly by address (e.g.
  axonpro_int8_packing_filter) - fixed lookup tables baked into the Axon driver library itself
  (drivers/axon), not the deployed application's own code, but still only resolvable once that
  library has actually been linked into something.
- axon_model_<name>_persistent_vars (RAM backing store for a model's persistent variables, if
  any - sized per-model, so the application declares one array per model instance it hosts).

The first three are already declared `extern` by headers the model header itself pulls in
(nrf_axon_nn_infer.h / nrf_axon_nn_op_extensions.h / nrf_axon_driver.h), so referencing them from
a standalone compile unit that never defines them is fine as-is - the model stub link just needs
to resolve them to real addresses via a generated PROVIDE() linker script (see
extract_elf_syms.py).

persistent_vars is different: the generated header defines the array itself (plain
`int32_t axon_model_<name>_persistent_vars[N];`, no `extern`), because ordinarily this header is
compiled straight into the application that owns it. The model stub link must not define its own
copy of that array - it needs to reference the *application's* copy - so this script rewrites
that one declaration line to `extern` in a patched copy of the header, which the model stub
compiles instead of the original. Nothing else in the header's text is touched.
"""
import argparse
import re
import sys
from pathlib import Path

OP_EXTENSION_RE = re.compile(r"\bnrf_axon_nn_op_extension_\w+\b")
AXONPRO_CONST_RE = re.compile(r"\baxonpro_\w+\b")
PERSISTENT_VARS_DEF_RE = re.compile(
	r"^(?P<indent>[ \t]*)(?P<type>u?int(?:8|16|32|64)_t)\s+"
	r"(?P<name>axon_model_\w+_persistent_vars)\s*\[\s*(?P<size>\d+)\s*\]\s*;",
	re.MULTILINE)

INTERLAYER_BUFFER_SYMBOL = "nrf_axon_interlayer_buffer"


def find_op_extension_symbols(header_text):
	return sorted(set(OP_EXTENSION_RE.findall(header_text)))


def find_axonpro_const_symbols(header_text):
	"""Driver-owned lookup-table constants (e.g. axonpro_int8_packing_filter) some cmd_buffer
	entries reference directly by address - see the module docstring."""
	return sorted(set(AXONPRO_CONST_RE.findall(header_text)))


def find_persistent_vars_symbols(header_text):
	"""Returns [(name, type, size), ...] for every persistent_vars array *definition* (not
	just any mention of the name, which also appears in cmd_buffer/pointer-arithmetic
	expressions elsewhere in the header) found in header_text."""
	return [(m.group("name"), m.group("type"), int(m.group("size")))
		for m in PERSISTENT_VARS_DEF_RE.finditer(header_text)]


def collect_app_symbols(header_text):
	"""Returns the sorted, de-duplicated list of every app-owned symbol name this header
	references, for nrf_axon_model_stub() (see nrf_axon_model_stub.cmake) to force-undefined
	and extract real addresses for. Pure function - no filesystem access - so it can be
	unit-tested directly."""
	symbols = set(find_op_extension_symbols(header_text))
	symbols.add(INTERLAYER_BUFFER_SYMBOL)
	symbols.update(find_axonpro_const_symbols(header_text))
	symbols.update(name for name, _type, _size in find_persistent_vars_symbols(header_text))
	return sorted(symbols)


def patch_persistent_vars_definitions(header_text):
	"""Returns header_text with every persistent_vars array *definition* rewritten to an
	`extern` declaration of the same name/type/size - see the module docstring for why. Pure
	function - no filesystem access - so it can be unit-tested directly."""
	def replace(match):
		return "%sextern %s %s[%s];" % (
			match.group("indent"), match.group("type"), match.group("name"),
			match.group("size"))

	return PERSISTENT_VARS_DEF_RE.sub(replace, header_text)


def main():
	parser = argparse.ArgumentParser(description=__doc__,
					  formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--header", type=Path, required=True,
			     help="Path to the compiler-generated model header")
	parser.add_argument("--print-symbols", action="store_true",
			     help="Print discovered app-owned symbol names, one per line, and exit")
	parser.add_argument("--symbols-out", type=Path, default=None,
			     help="Write discovered app-owned symbol names, one per line")
	parser.add_argument("--patched-header-out", type=Path, default=None,
			     help="Write a copy of --header with persistent_vars definitions "
				  "rewritten to extern declarations")
	args = parser.parse_args()

	header_text = args.header.read_text(encoding="utf-8")
	symbols = collect_app_symbols(header_text)

	if args.print_symbols:
		print("\n".join(symbols))
		return 0

	if args.symbols_out is None or args.patched_header_out is None:
		parser.error("--symbols-out and --patched-header-out are required unless "
			      "--print-symbols is set")

	args.symbols_out.write_text("\n".join(symbols) + "\n", encoding="utf-8")
	args.patched_header_out.write_text(patch_persistent_vars_definitions(header_text),
					     encoding="utf-8")
	return 0


if __name__ == "__main__":
	sys.exit(main())
