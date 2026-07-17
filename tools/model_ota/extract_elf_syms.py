#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Extract symbol addresses from a deployed application's zephyr.elf and emit a linker-script
fragment of PROVIDE() entries for them - the second half of the "second-pass link" model stub
mechanism (see nrf_axon_model_stub.cmake and gen_axon_stub_fixups.py for the first half).

This is what lets a model stub, linked completely separately from the application, still resolve
app-owned symbols (nrf_axon_interlayer_buffer, nrf_axon_nn_op_extension_*,
axon_model_<name>_persistent_vars) to the addresses the *deployed* application actually placed
them at - not a separate "reference build"'s addresses, which are not guaranteed to match.

PROVIDE() only takes effect for a symbol that would otherwise be undefined at link time, which is
exactly the state every symbol in --symbols is in for the model stub (see
gen_axon_stub_fixups.py's patched header for why persistent_vars arrays are extern there).
"""
import argparse
import subprocess
import sys
from pathlib import Path

OP_EXTENSION_PREFIX = "nrf_axon_nn_op_extension_"


def lookup_symbol(nm, elf, symbol):
	output = subprocess.check_output([nm, str(elf)], text=True, errors="replace")
	for line in output.splitlines():
		parts = line.split()
		if len(parts) >= 3 and parts[-1] == symbol:
			return int(parts[0], 16)
	return None


def build_provide_script(nm, elf, symbols):
	"""Returns (linker_script_text, missing_symbols). Thumb code pointers (op extension
	functions) get bit 0 set in the stored address, matching how the compiler itself encodes
	a function pointer constant for a Cortex-M target."""
	lines = []
	missing = []
	for symbol in symbols:
		addr = lookup_symbol(nm, elf, symbol)
		if addr is None:
			missing.append(symbol)
			continue
		if symbol.startswith(OP_EXTENSION_PREFIX):
			addr |= 1
		lines.append("PROVIDE(%s = 0x%X);" % (symbol, addr))
	return "\n".join(lines) + "\n", missing


def main():
	parser = argparse.ArgumentParser(description=__doc__,
					  formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument("--nm", required=True, help="Path to the toolchain's nm")
	parser.add_argument("--elf", type=Path, required=True,
			     help="Deployed application's zephyr.elf (already built, never a "
				  "separate reference build)")
	parser.add_argument("--symbols", type=Path, required=True,
			     help="Text file with one app-owned symbol name per line "
				  "(see gen_axon_stub_fixups.py --symbols-out)")
	parser.add_argument("--output", type=Path, required=True,
			     help="Where to write the generated PROVIDE() linker script")
	args = parser.parse_args()

	symbols = [line.strip() for line in args.symbols.read_text(encoding="utf-8").splitlines()
		   if line.strip() and not line.strip().startswith("#")]

	script_text, missing = build_provide_script(args.nm, args.elf, symbols)

	if missing:
		print("error: symbol(s) not found in %s: %s" % (args.elf, ", ".join(missing)),
		      file=sys.stderr)
		print("(app-owned symbols referenced by a model header must be force-kept alive "
		      "in the deployed application - see toolchain_ld_force_undefined_symbols() "
		      "in nrf_axon_model_stub.cmake)", file=sys.stderr)
		return 1

	args.output.write_text(script_text, encoding="ascii")
	return 0


if __name__ == "__main__":
	sys.exit(main())
