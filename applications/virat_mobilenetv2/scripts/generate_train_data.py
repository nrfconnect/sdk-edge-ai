#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Generates a minimal representative dataset .npy for Keras->TFLite conversion.
Loads the model to get input shape, then fills with random values in [0,1].
For real quantization quality, replace with actual training/calibration images.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def get_keras_input_shape(keras_path: str) -> tuple:
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("Need tensorflow. Install with: pip install tensorflow")
    # compile=False avoids deserializing custom loss (only need input shape)
    model = tf.keras.models.load_model(keras_path, compile=False)
    shape = model.input_shape
    if len(shape) != 4:
        raise ValueError(f"Expected 4D input (batch, H, W, C), got {shape}")
    return tuple(int(s) for s in shape[1:])  # (H, W, C)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate x_train_virat.npy for Axon compiler")
    ap.add_argument(
        "--model",
        default="models/virat_mobilenetv2.keras",
        help="Path to .keras or .h5 model",
    )
    ap.add_argument(
        "--out",
        default="data/x_train_virat.npy",
        help="Output .npy path",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Number of samples (default 500)",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    model_path = root / args.model
    out_path = root / args.out
    if not model_path.is_file():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1
    h, w, c = get_keras_input_shape(str(model_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Representative data: float in [0, 1]
    x_train = np.random.uniform(0.0, 1.0, (args.samples, h, w, c)).astype(np.float32)
    np.save(str(out_path), x_train)
    print(f"Saved shape {x_train.shape} to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
