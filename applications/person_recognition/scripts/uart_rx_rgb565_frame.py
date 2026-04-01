#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Receive one binary RGB565 frame from person_recognition (CONFIG_PERSON_RECOGNITION_DUMP_UART_FRAME).

Default camera mode is 128x128 RGB565 (payload 32768 bytes). Override with --expect-w/--expect-h.

Frame format (little-endian, after optional leading log text):
  magic: 4 bytes  A5 5A 34 12
  width: uint16
  height: uint16
  payload_len: uint32
  crc32_ieee: uint32  (same as zlib.crc32(payload) & 0xFFFFFFFF)
  payload: payload_len bytes (RGB565 big-endian per pixel, row-major)
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib

try:
    import serial
except ImportError as e:
    raise SystemExit("pyserial required: pip install pyserial") from e

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("Pillow required: pip install Pillow") from e

MAGIC = bytes((0xA5, 0x5A, 0x34, 0x12))

# Matches person_recognition CAM_W x CAM_H when using 128x128 capture
DEFAULT_EXPECT_W = 128
DEFAULT_EXPECT_H = 128


def rgb565_be_to_rgb888(buf: bytes, width: int, height: int) -> Image.Image:
    expected = width * height * 2
    if len(buf) != expected:
        raise ValueError(f"payload length {len(buf)} != {expected} for {width}x{height} RGB565")
    out = bytearray(width * height * 3)
    o = 0
    for i in range(0, len(buf), 2):
        hi, lo = buf[i], buf[i + 1]
        pix = (hi << 8) | lo
        r5 = (pix >> 11) & 0x1F
        g6 = (pix >> 5) & 0x3F
        b5 = pix & 0x1F
        out[o + 0] = (r5 << 3) | (r5 >> 2)
        out[o + 1] = (g6 << 2) | (g6 >> 4)
        out[o + 2] = (b5 << 3) | (b5 >> 2)
        o += 3
    return Image.frombytes("RGB", (width, height), bytes(out))


def validate_rgb_image(img: Image.Image) -> tuple[bool, str]:
    """Return (ok, message) — flag flat / degenerate images."""
    extrema = img.getextrema()
    if not extrema:
        return False, "empty image"
    ranges = [mx - mn for mn, mx in extrema]
    rmax = max(ranges)
    if rmax == 0:
        return False, "flat image (single color — camera or bus may be stuck)"
    return True, f"pixel range OK (max channel span {rmax})"


def find_magic(stream: bytes) -> int:
    return stream.find(MAGIC)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Read one person_recognition RGB565 UART frame and save PNG (128x128 by default)"
    )
    ap.add_argument("--port", default="/dev/ttyACM0", help="Serial device")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", type=str, default="uart_frame.png")
    ap.add_argument(
        "--expect-w",
        type=int,
        default=DEFAULT_EXPECT_W,
        metavar="W",
        help=f"Expected width (0 = skip check, default {DEFAULT_EXPECT_W})",
    )
    ap.add_argument(
        "--expect-h",
        type=int,
        default=DEFAULT_EXPECT_H,
        metavar="H",
        help=f"Expected height (0 = skip check, default {DEFAULT_EXPECT_H})",
    )
    ap.add_argument(
        "--no-content-check",
        action="store_true",
        help="Do not warn/fail on flat single-color images",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for frame after opening port",
    )
    args = ap.parse_args()

    import time

    ser = serial.Serial(args.port, args.baud, timeout=0.5)
    buf = b""
    deadline = time.monotonic() + args.timeout

    while time.monotonic() < deadline:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
            idx = find_magic(buf)
            if idx >= 0:
                buf = buf[idx:]
                break
        time.sleep(0.02)
    else:
        print("Timeout waiting for magic", file=sys.stderr)
        ser.close()
        return 1

    while len(buf) < 16 and time.monotonic() < deadline:
        buf += ser.read(16 - len(buf))

    if len(buf) < 16:
        print("Timeout reading header", file=sys.stderr)
        ser.close()
        return 1

    if buf[:4] != MAGIC:
        print("Bad magic", file=sys.stderr)
        ser.close()
        return 1

    w, h, plen, crc_expect = struct.unpack_from("<HHII", buf, 4)
    if plen != w * h * 2:
        print(f"Unexpected payload_len {plen} for {w}x{h}", file=sys.stderr)
        ser.close()
        return 1

    payload = bytes(buf[16:]) if len(buf) > 16 else b""
    if len(payload) > plen:
        payload = payload[:plen]
    while len(payload) < plen and time.monotonic() < deadline:
        payload += ser.read(plen - len(payload))

    if len(payload) != plen:
        print("Timeout reading payload", file=sys.stderr)
        ser.close()
        return 1

    crc_got = zlib.crc32(payload) & 0xFFFFFFFF
    if crc_got != crc_expect:
        print(f"CRC mismatch: got {crc_got:08x} expect {crc_expect:08x}", file=sys.stderr)
        ser.close()
        return 1

    if args.expect_w and w != args.expect_w:
        print(
            f"Width mismatch: got {w} expected {args.expect_w} (use --expect-w 0 to allow any)",
            file=sys.stderr,
        )
        ser.close()
        return 1
    if args.expect_h and h != args.expect_h:
        print(
            f"Height mismatch: got {h} expected {args.expect_h} (use --expect-h 0 to allow any)",
            file=sys.stderr,
        )
        ser.close()
        return 1

    img = rgb565_be_to_rgb888(payload, w, h)

    if not args.no_content_check:
        ok, msg = validate_rgb_image(img)
        print(msg)
        if not ok:
            ser.close()
            return 1

    img.save(args.out)
    print(f"OK: wrote {args.out} ({w}x{h} RGB from RGB565, {plen} bytes, CRC verified)")
    ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
