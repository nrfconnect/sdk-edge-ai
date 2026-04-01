#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Converts test pictures to model input format (int8) and emits a C header.
Supports presets (vww, virat) or --axon-header to match a compiled Axon model.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

# Default quantization (tinyml_vww / many int8 models); overridden by --axon-header
INPUT_QUANT_MULT = 133693432
INPUT_QUANT_ROUND = 19
INPUT_QUANT_ZP = -128

# Model presets: (height, width, chans, use_chw)
MODELS = {
    "vww": (96, 96, 3, True),
    "virat": (360, 640, 3, False),
}


def parse_axon_input_from_header(header_path: Path, model_symbol: str) -> dict:
    """
    Read external input layout from nrf_axon_model_*.h (first input block after model symbol).
    Returns: height, width, channel_cnt, quant_mult, quant_round, quant_zp, use_chw
    """
    text = header_path.read_text(encoding="utf-8", errors="replace")
    anchor = f"const nrf_axon_nn_compiled_model_s {model_symbol}"
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError(f"Could not find '{anchor}' in {header_path}")
    chunk = text[idx : idx + 12000]
    dim_m = re.search(
        r"\.dimensions\s*=\s*\{\s*"
        r"\.height\s*=\s*(\d+),\s*"
        r"\.width\s*=\s*(\d+),\s*"
        r"\.channel_cnt\s*=\s*(\d+),",
        chunk,
        re.DOTALL,
    )
    if not dim_m:
        raise ValueError(f"Could not parse .dimensions in {header_path}")
    height, width, chans = (int(dim_m.group(1)), int(dim_m.group(2)), int(dim_m.group(3)))
    qm_m = re.search(r"\.quant_mult\s*=\s*(\d+),", chunk)
    qr_m = re.search(r"\.quant_round\s*=\s*(\d+),", chunk)
    zp_m = re.search(r"\.quant_zp\s*=\s*(-?\d+),", chunk)
    st_m = re.search(r"\.stride\s*=\s*(\d+),", chunk)
    if not all([qm_m, qr_m, zp_m, st_m]):
        raise ValueError(f"Could not parse quant/stride in {header_path}")
    stride = int(st_m.group(1))
    # Stride == width: channel planes (CHW). Stride == width*chans: interleaved HWC.
    if stride == width:
        use_chw = True
    elif stride == width * chans:
        use_chw = False
    else:
        use_chw = True
    return {
        # "height": height,
        # "width": width,
        "height": 144,
        "width": 144,
        "channel_cnt": chans,
        "quant_mult": int(qm_m.group(1)),
        "quant_round": int(qr_m.group(1)),
        "quant_zp": int(zp_m.group(1)),
        "use_chw": use_chw,
    }


def load_and_quantize(
    image_path: Path,
    height: int,
    width: int,
    use_chw: bool,
    quant_mult: int,
    quant_round: int,
    quant_zp: int,
    preprocess: str,
) -> np.ndarray:
    """
    preprocess:
      unit01 — float in [0, 1] (typical ImageNet-style int8 models, e.g. tinyml_vww).
      symmetric_m1_1 — (pixel/255)*2 - 1 in [-1, 1], same as machine_learning/mcunet/eval_det.py
        for MCUNet VWW TFLite (input quant scale ~1/127, zp -1).
    """
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("PIL required. Install with: pip install Pillow")
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    x = np.array(img, dtype=np.float32) / 255.0
    if preprocess == "symmetric_m1_1":
        x = x * 2.0 - 1.0
    elif preprocess != "unit01":
        raise ValueError(f"unknown preprocess: {preprocess}")
    q = (x * quant_mult).astype(np.int64)
    q = (q >> quant_round) + quant_zp
    q = np.clip(q, -128, 127).astype(np.int8)
    if use_chw:
        q = np.transpose(q, (2, 0, 1))
    return q


def array_to_c(name: str, arr: np.ndarray) -> str:
    flat = arr.flatten()
    lines = ["static const int8_t " + name + "[" + str(len(flat)) + "] = {"]
    for i in range(0, len(flat), 16):
        chunk = flat[i : i + 16]
        lines.append("  " + ", ".join(str(int(x)) for x in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert test pictures to model input C header")
    ap.add_argument(
        "--axon-header",
        default=None,
        help="Path to nrf_axon_model_*.h (uses input dims + quantization from compiled model)",
    )
    ap.add_argument(
        "--model-symbol",
        default="model_mcunet_vww_320kb",
        help="C symbol after nrf_axon_nn_compiled_model_s (default: model_mcunet_vww_320kb)",
    )
    ap.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="vww",
        help="Preset when --axon-header is not set",
    )
    ap.add_argument(
        "--pictures-dir",
        default="pictures",
        help="Directory containing demo_picture.jpeg, demo_2.jpeg, demo_3.jpeg",
    )
    ap.add_argument(
        "--out",
        default="src/generated/test_images.h",
        help="Output header path (default: src/generated/test_images.h)",
    )
    ap.add_argument(
        "--images",
        nargs="+",
        metavar="FILE",
        default=["demo_picture.jpeg", "demo_2.jpeg", "demo_3.jpeg"],
        help="JPEG filenames under pictures-dir (default: three demos). "
        "Use e.g. --images demo_picture.jpeg to save flash.",
    )
    ap.add_argument(
        "--preprocess",
        choices=("unit01", "symmetric_m1_1"),
        default="unit01",
        help="Float range before Axon quant: unit01=[0,1]; symmetric_m1_1=[-1,1] (MCUNet eval_det.py). "
        "Use symmetric_m1_1 for mcunet VWW int8 models.",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    pics_dir = root / args.pictures_dir
    out_path = root / args.out

    if args.axon_header:
        hdr = Path(args.axon_header)
        if not hdr.is_file():
            print(f"Error: axon header not found: {hdr}", file=sys.stderr)
            return 1
        try:
            spec = parse_axon_input_from_header(hdr, args.model_symbol)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        height = spec["height"]
        width = spec["width"]
        use_chw = spec["use_chw"]
        qm, qr, qz = spec["quant_mult"], spec["quant_round"], spec["quant_zp"]
    else:
        height, width, _chans, use_chw = MODELS[args.model]
        qm, qr, qz = INPUT_QUANT_MULT, INPUT_QUANT_ROUND, INPUT_QUANT_ZP

    arrays = []
    for filename in args.images:
        path = pics_dir / filename
        if not path.is_file():
            print(f"Error: image not found: {path}", file=sys.stderr)
            return 1
        var_name = Path(filename).stem.replace("-", "_")
        arr = load_and_quantize(path, height, width, use_chw, qm, qr, qz, args.preprocess)
        arrays.append((var_name, arr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "/* Auto-generated by scripts/embed_test_images.py */",
        "#ifndef PERSON_RECOGNITION_TEST_IMAGES_H",
        "#define PERSON_RECOGNITION_TEST_IMAGES_H",
        "",
        "#include <stdint.h>",
        "",
        "#define PERSON_RECOGNITION_NUM_TEST_IMAGES " + str(len(arrays)),
        "",
    ]
    for var_name, arr in arrays:
        parts.append(array_to_c("test_image_" + var_name, arr))
        parts.append("")
    parts.append("static const int8_t *const person_recognition_test_inputs[] = {")
    for var_name, _ in arrays:
        parts.append("  test_image_" + var_name + ",")
    parts.append("};")
    parts.append("")
    parts.append('static const char *const person_recognition_test_names[] = {')
    for var_name, _ in arrays:
        parts.append('  "' + var_name + '",')
    parts.append("};")
    parts.append("")
    parts.append("#endif /* PERSON_RECOGNITION_TEST_IMAGES_H */")
    out_path.write_text("\n".join(parts))
    print(f"Wrote {out_path} ({len(arrays)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
