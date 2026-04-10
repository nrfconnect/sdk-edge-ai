#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Compare TFLite input/output metadata with nrf_axon_model_*.h (H/W, dtypes).
Run from person_recognition after embedding issues or re-exporting .tflite.

  python3 scripts/check_tflite_vs_axon.py \\
    --tflite models/mcunet-320kb-1mb_vww.tflite \\
    --axon-header outputs/nrf_axon_model_mcunet_vww_320kb_.h \\
    --model-symbol model_mcunet_vww_320kb
"""
import argparse
import re
import sys
from pathlib import Path


def parse_axon_dims(header_path: Path, model_symbol: str):
    text = header_path.read_text(encoding="utf-8", errors="replace")
    anchor = f"const nrf_axon_nn_compiled_model_s {model_symbol}"
    idx = text.find(anchor)
    if idx < 0:
        raise SystemExit(f"model symbol not found: {model_symbol}")
    chunk = text[idx : idx + 12000]
    dim_m = re.search(
        r"\.dimensions\s*=\s*\{\s*\.height\s*=\s*(\d+),\s*\.width\s*=\s*(\d+),",
        chunk,
    )
    if not dim_m:
        raise SystemExit("could not parse input dimensions")
    return int(dim_m.group(1)), int(dim_m.group(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", required=True)
    ap.add_argument("--axon-header", required=True)
    ap.add_argument("--model-symbol", default="model_mcunet_vww_320kb")
    args = ap.parse_args()
    tflite_path = Path(args.tflite)
    hdr_path = Path(args.axon_header)
    if not tflite_path.is_file():
        print(f"Missing tflite: {tflite_path}", file=sys.stderr)
        return 1
    if not hdr_path.is_file():
        print(f"Missing header: {hdr_path}", file=sys.stderr)
        return 1

    try:
        import tensorflow as tf
    except ImportError:
        print("Install tensorflow to run this check.", file=sys.stderr)
        return 1

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    shp = inp["shape"]
    if len(shp) != 4:
        print("Unexpected input rank:", shp, file=sys.stderr)
        return 1
    _, h_tf, w_tf, c_tf = [int(x) for x in shp]

    ah, aw = parse_axon_dims(hdr_path, args.model_symbol)
    print("TFLite input shape (NHWC):", [int(x) for x in shp], "dtype", inp["dtype"])
    print("TFLite input quantization:", inp.get("quantization"))
    print("TFLite output shape:", [int(x) for x in out["shape"]], "dtype", out["dtype"])
    print("TFLite output quantization:", out.get("quantization"))
    print(f"Axon header input HxW: {ah}x{aw}")
    if h_tf != ah or w_tf != aw:
        print(
            "*** MISMATCH: firmware uses Axon HxW; this .tflite differs. "
            "Re-run the Axon compiler on this exact .tflite (or restore the .tflite used for compile).",
            file=sys.stderr,
        )
        return 2
    print("OK: spatial dimensions match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
