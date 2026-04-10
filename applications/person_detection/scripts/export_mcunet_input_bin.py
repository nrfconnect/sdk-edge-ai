#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Export raw int8 CHW input (Axon packed layout) for mcunet_vww_axon_sim.

  python3 scripts/export_mcunet_input_bin.py \\
    --jpeg pictures/demo_picture.jpeg \\
    --out /tmp/mcunet_in.bin
"""
import argparse
import importlib.util
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent


def _load_embed():
    spec = importlib.util.spec_from_file_location(
        "embed_test_images", SCRIPT_DIR / "embed_test_images.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jpeg", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--axon-header",
        type=Path,
        default=APP_DIR / "outputs/nrf_axon_model_mcunet_vww_320kb_.h",
    )
    ap.add_argument("--model-symbol", default="model_mcunet_vww_320kb")
    args = ap.parse_args()

    if not args.axon_header.is_file():
        print(f"Missing {args.axon_header}", file=sys.stderr)
        return 1
    if not args.jpeg.is_file():
        print(f"Missing {args.jpeg}", file=sys.stderr)
        return 1

    embed = _load_embed()
    spec = embed.parse_axon_input_from_header(args.axon_header, args.model_symbol)
    arr = embed.load_and_quantize(
        args.jpeg,
        spec["height"],
        spec["width"],
        spec["use_chw"],
        spec["quant_mult"],
        spec["quant_round"],
        spec["quant_zp"],
        "symmetric_m1_1",
    )
    raw = arr.tobytes(order="C")
    args.out.write_bytes(raw)
    print(f"Wrote {len(raw)} bytes to {args.out} ({spec['height']}x{spec['width']} CHW int8)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
