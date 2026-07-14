#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Shared model_storage partition layout helpers for the model-only OTA update PoC's
packaging tools (package_model.py, package_model_axon.py).

Both tools need to know the model_storage partition's flash address and size: the address to
addresses the .hex file (and, for Axon, to bake into cmd_buffer's relocated pointers), and both
to preflight-check that the package they are about to write will actually fit, instead of only
finding that out after flashing a truncated/overlapping image.

Rather than trusting a hand-typed --address/--partition-size to still match a board's overlay,
--dts lets both tools read them straight out of a build's generated zephyr.dts, which is the
source of truth for where model_storage actually is.
"""
import re

# nRF54LM20 DK 'model_storage' partition (see samples/*/boards/*.overlay), used as the default
# when neither --dts nor an explicit --address/--partition-size is given.
DEFAULT_ADDRESS = 0x102000
DEFAULT_PARTITION_SIZE = 968 * 1024

_PARTITION_NODE_RE = re.compile(
	r'partition@[0-9a-fA-F]+\s*\{'
	r'(?P<body>(?:[^{}]|\{[^{}]*\})*?)'
	r'\};',
	re.S,
)
_LABEL_RE = re.compile(r'label\s*=\s*"model_storage"\s*;')
_REG_RE = re.compile(r'reg\s*=\s*<\s*(0[xX][0-9a-fA-F]+|\d+)\s+(0[xX][0-9a-fA-F]+|\d+)\s*>\s*;')


def read_partition_layout_from_dts(dts_path):
	"""Returns (address, size) for the 'model_storage' partition in a generated zephyr.dts.

	Raises ValueError if no partition node with label = "model_storage" and a `reg = <addr
	size>;` property can be found."""
	text = dts_path.read_text()

	for match in _PARTITION_NODE_RE.finditer(text):
		body = match.group("body")
		if not _LABEL_RE.search(body):
			continue
		reg = _REG_RE.search(body)
		if not reg:
			raise ValueError(
				"Found the 'model_storage' partition node in %s, but could not "
				"parse its reg = <addr size>; property" % dts_path)
		return int(reg.group(1), 0), int(reg.group(2), 0)

	raise ValueError("Could not find a partition node with label = \"model_storage\" in %s - "
			  "is this the right board's generated zephyr.dts?" % dts_path)


def check_package_fits(total_size, partition_size, source_description):
	"""Raises ValueError if a total_size-byte package would not fit in partition_size bytes
	of model_storage, so an oversized package is caught here instead of after flashing it."""
	if total_size > partition_size:
		raise ValueError(
			"Package (%u B, from %s) exceeds model_storage partition capacity "
			"(%u B). Provide a larger --partition-size (or --dts pointing at a "
			"build with a bigger model_partition overlay), or reduce the model."
			% (total_size, source_description, partition_size))


def format_size(size):
	"""Formats a byte count as whole KiB when it divides evenly, otherwise as bytes -
	mirrors the style of Zephyr's own "Memory region" build-time usage table."""
	if size >= 1024 and size % 1024 == 0:
		return "%u KB" % (size // 1024)
	return "%u B" % size


def report_package_usage(total_size, partition_size, label="model_storage"):
	"""Prints a one-line utilization summary for a package against the partition it will be
	flashed into, e.g.:

	    Partition region      Used Size  Region Size  %age Used
	            model_storage:      42 KB      968 KB      4.34%

	Does not raise on overflow - call check_package_fits() for that; this is purely a
	developer-facing "how much headroom do I have" report, printed on every successful
	build so shrinking headroom is visible long before a model actually stops fitting."""
	percent = (total_size * 100.0) / partition_size if partition_size else float("inf")
	print("Partition region      Used Size  Region Size  %age Used")
	print("%21s: %10s %10s %8.2f%%" %
	      (label, format_size(total_size), format_size(partition_size), percent))
