#!/usr/bin/env python3
#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
"""Patch the CRC32 field of a linked Neuton or Axon model partition image.

The image binary is produced by the compiler/linker + `objcopy -O binary -j .model_image`, with
the header's crc32 field left as 0. This tool computes CRC-32/IEEE (== zlib.crc32, == Zephyr's
crc32_ieee) over the whole image with the crc32 field held at 0, and writes the result back into
that field. The on-device loaders (model_image_load_neuton() / model_image_load_axon())
recompute the CRC exactly the same way, so the model *data* remains entirely
compiler/linker-produced - only these 4 CRC bytes are written by the host.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

# Must match MODEL_IMAGE_CRC32_OFFSET in include/model_ota/model_image.h.
DEFAULT_CRC_OFFSET = 12


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, required=True,
                        help="Raw image binary (crc32 field expected to be 0)")
    parser.add_argument("-o", "--out", type=Path, required=True,
                        help="Output path for the crc-patched image binary")
    parser.add_argument("--crc-offset", type=lambda x: int(x, 0), default=DEFAULT_CRC_OFFSET,
                        help="Byte offset of the crc32 field (default %d)" % DEFAULT_CRC_OFFSET)
    args = parser.parse_args()

    data = bytearray(args.bin.read_bytes())
    if len(data) < args.crc_offset + 4:
        parser.error("image (%u B) is smaller than the crc32 field offset" % len(data))

    # Hold the crc field at 0 while computing, exactly as the loader does.
    struct.pack_into("<I", data, args.crc_offset, 0)
    crc = zlib.crc32(data) & 0xFFFFFFFF
    struct.pack_into("<I", data, args.crc_offset, crc)

    args.out.write_bytes(data)
    print("Patched CRC32 0x%08x over %u B image -> %s" % (crc, len(data), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
