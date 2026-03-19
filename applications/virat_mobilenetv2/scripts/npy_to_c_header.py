#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Converts a compiler .npy output (e.g. quantized test data) to a C header
with one int8 array. Use when you have the model .h from a successful compile
and only need to embed a single input vector from the .npy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert .npy (int8) to C header with one array")
    ap.add_argument("--npy", required=True, help="Path to .npy file")
    ap.add_argument("--slice", type=int, default=0, help="Index of sample to export (default 0)")
    ap.add_argument("--out", default="src/input_vector.h", help="Output .h path")
    ap.add_argument("--name", default="input_vector", help="C array name")
    args = ap.parse_args()
    npy_path = Path(args.npy)
    out_path = Path(args.out)
    if not npy_path.is_file():
        print(f"Error: not found: {npy_path}", file=sys.stderr)
        return 1
    arr = np.load(npy_path)
    if arr.ndim > 1:
        arr = arr[args.slice]
    arr = np.asarray(arr, dtype=np.int8).flatten()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "/* Auto-generated from " + npy_path.name + " */",
        "#ifndef NPY_INPUT_VECTOR_H",
        "#define NPY_INPUT_VECTOR_H",
        "",
        "#include <stdint.h>",
        "",
        f"#define {args.name.upper()}_LEN " + str(len(arr)),
        "",
        "static const int8_t " + args.name + "[" + str(len(arr)) + "] = {",
    ]
    for i in range(0, len(arr), 16):
        chunk = arr[i : i + 16]
        lines.append("  " + ", ".join(str(int(x)) for x in chunk) + ",")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path} (size {len(arr)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
