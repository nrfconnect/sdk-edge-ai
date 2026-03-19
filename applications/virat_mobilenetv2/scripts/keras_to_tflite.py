#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Converts .keras (or .h5) to int8-quantized .tflite using a representative dataset.
Run this first, then run the Axon compiler on the generated .tflite.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def convert(keras_path: str, train_npy_path: str, out_tflite_path: str) -> None:
    try:
        import tensorflow as tf
    except ImportError:
        raise RuntimeError("Need tensorflow. Install with: pip install tensorflow")

    x_train = np.load(train_npy_path).astype(np.float32)
    if x_train.ndim == 3:
        x_train = np.expand_dims(x_train, axis=0)

    if keras_path.endswith(".h5") or keras_path.endswith(".keras"):
        model = tf.keras.models.load_model(keras_path, compile=False)
        if model.input_shape[1:]:
            target_shape = tuple(int(s) for s in model.input_shape[1:])
            if x_train.shape[1:] != target_shape:
                x_train = x_train.reshape((x_train.shape[0],) + target_shape)
        # Use concrete function to avoid _get_save_spec and _DictWrapper issues (Keras 3 / TF 2.16+)
        # Fixed batch size 1 for TFLite compatibility
        in_shape = model.input_shape
        input_shape = (1,) + tuple(int(s) for s in in_shape[1:])
        input_spec = tf.TensorSpec(shape=input_shape, dtype=tf.float32)

        @tf.function(input_signature=[input_spec])
        def serve(inp):
            return model(inp)

        concrete_func = serve.get_concrete_function()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    else:
        converter = tf.lite.TFLiteConverter.from_saved_model(keras_path)

    def representative_dataset_gen():
        n = max(1, int(0.1 * x_train.shape[0]))
        for i in range(n):
            yield [np.expand_dims(x_train[i], axis=0).astype(np.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    Path(out_tflite_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_tflite_path).write_bytes(tflite_model)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert .keras/.h5 to int8 .tflite")
    ap.add_argument(
        "--keras",
        default="models/virat_mobilenetv2.keras",
        help="Path to .keras or .h5 model",
    )
    ap.add_argument(
        "--train-data",
        default="data/x_train_virat.npy",
        help="Path to representative dataset .npy (shape N,H,W,C float32)",
    )
    ap.add_argument(
        "--out",
        default="models/virat_mobilenetv2.tflite",
        help="Output .tflite path",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    keras_path = root / args.keras
    train_path = root / args.train_data
    out_path = root / args.out
    if not keras_path.is_file():
        print(f"Error: model not found: {keras_path}", file=sys.stderr)
        return 1
    if not train_path.is_file():
        print(f"Error: train data not found: {train_path}", file=sys.stderr)
        return 1
    try:
        convert(str(keras_path), str(train_path), str(out_path))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
