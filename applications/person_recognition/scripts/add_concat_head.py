#!/usr/bin/env python3
"""
add_concat_head.py
==================
Rewrites models/person-det.tflite so that the three DEQUANTIZE output heads
are merged into a single [1, 420, 18] float32 tensor via RESHAPE x3 + CONCAT.

Existing terminal outputs (float32, produced by the 3 DEQUANTIZE ops):
  tensor  11  –  [1,  4,  5, 18]   (small  head, 20 detections)
  tensor 108  –  [1,  8, 10, 18]   (medium head, 80 detections)
  tensor 153  –  [1, 16, 20, 18]   (large  head, 320 detections)

New graph tail appended:
  RESHAPE(11  → [1,  20, 18])
  RESHAPE(108 → [1,  80, 18])
  RESHAPE(153 → [1, 320, 18])
  CONCAT(axis=1 → [1, 420, 18])   ← new single model output

Output written to:  models/person-det-concat.tflite

Usage:
  <venv>/python scripts/add_concat_head.py
"""

import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
COMPILER_SCRIPTS = ROOT.parent.parent / 'tools/axon/compiler/scripts'
sys.path.insert(0, str(COMPILER_SCRIPTS))

IN_MODEL  = ROOT / 'models/person-det.tflite'
OUT_MODEL = ROOT / 'models/person-det-concat.tflite'

# head tensor index → desired flat shape
HEADS = [
    (11,  [1,  20, 18]),   # small:  1×4×5×18  → 1×20×18
    (108, [1,  80, 18]),   # medium: 1×8×10×18 → 1×80×18
    (153, [1, 320, 18]),   # large:  1×16×20×18 → 1×320×18
]
OUT_SHAPE = [1, 420, 18]


# ── probe which mutable-schema API is available ──────────────────────────────
def _probe():
    # TF 2.x — schema_py_generated lives under tensorflow.lite.python,
    # flatbuffer_utils (with the convert_* helpers) lives under tensorflow.lite.tools
    try:
        from tensorflow.lite.python import schema_py_generated as sfb
        from tensorflow.lite.tools import flatbuffer_utils as fbu
        _ = sfb.ModelT()
        assert callable(fbu.convert_bytearray_to_object)
        return sfb, fbu
    except Exception as e:
        print(f"  schema_py_generated prob failed: {e}")

    # Older TF layout (schema_fb + python flatbuffer_utils)
    try:
        from tensorflow.lite.python import schema_fb as sfb
        from tensorflow.lite.python import flatbuffer_utils as fbu
        _ = sfb.ModelT()
        return sfb, fbu
    except Exception as e:
        print(f"  schema_fb probe failed: {e}")

    return None, None


# ── model mutation ────────────────────────────────────────────────────────────
def build_model(sfb, fbu):
    raw = bytearray(IN_MODEL.read_bytes())

    # deserialise to mutable ModelT
    deser = (getattr(fbu, 'convert_bytearray_to_object', None) or
             getattr(fbu, 'read_model_with_mutable_tensors', None))
    if deser is None:
        raise RuntimeError("flatbuffer_utils has no known deserializer")
    model = deser(raw)

    sg        = model.subgraphs[0]
    op_codes  = model.operatorCodes

    # ── opcode helpers ────────────────────────────────────────────────────
    def get_or_add_opcode(builtin_code):
        for i, oc in enumerate(op_codes):
            if oc.builtinCode == builtin_code:
                return i
        oc = sfb.OperatorCodeT()
        oc.builtinCode            = builtin_code
        oc.deprecatedBuiltinCode  = min(builtin_code, 127)
        op_codes.append(oc)
        return len(op_codes) - 1

    reshape_oc = get_or_add_opcode(sfb.BuiltinOperator.RESHAPE)
    concat_oc  = get_or_add_opcode(sfb.BuiltinOperator.CONCATENATION)

    # ── tensor helpers ────────────────────────────────────────────────────
    def add_float_tensor(name, shape):
        t = sfb.TensorT()
        t.name         = name.encode()
        t.shape        = list(shape)
        t.type         = sfb.TensorType.FLOAT32
        t.buffer       = 0           # 0 = empty (runtime-allocated)
        t.quantization = sfb.QuantizationParametersT()
        sg.tensors.append(t)
        return len(sg.tensors) - 1

    def add_const_int32_tensor(name, data):
        arr = np.array(data, dtype=np.int32)
        buf = sfb.BufferT()
        buf.data = bytearray(arr.tobytes())
        model.buffers.append(buf)

        t = sfb.TensorT()
        t.name         = name.encode()
        t.shape        = list(arr.shape)
        t.type         = sfb.TensorType.INT32
        t.buffer       = len(model.buffers) - 1
        t.quantization = sfb.QuantizationParametersT()
        sg.tensors.append(t)
        return len(sg.tensors) - 1

    # ── append RESHAPE × 3 ────────────────────────────────────────────────
    reshaped_indices = []
    for src_idx, new_shape in HEADS:
        shape_t = add_const_int32_tensor(f'rs_shape_{src_idx}', new_shape)
        out_t   = add_float_tensor(f'rs_{src_idx}', new_shape)
        reshaped_indices.append(out_t)

        op = sfb.OperatorT()
        op.opcodeIndex = reshape_oc
        op.inputs      = [src_idx, shape_t]
        op.outputs     = [out_t]
        opts           = sfb.ReshapeOptionsT()
        opts.newShape  = list(new_shape)
        op.builtinOptionsType = sfb.BuiltinOptions.ReshapeOptions
        op.builtinOptions     = opts
        sg.operators.append(op)

    # ── append CONCAT ─────────────────────────────────────────────────────
    concat_out = add_float_tensor('detection_concat', OUT_SHAPE)

    op = sfb.OperatorT()
    op.opcodeIndex  = concat_oc
    op.inputs       = reshaped_indices
    op.outputs      = [concat_out]
    opts            = sfb.ConcatenationOptionsT()
    opts.axis       = 1
    opts.fusedActivationFunction = sfb.ActivationFunctionType.NONE
    op.builtinOptionsType = sfb.BuiltinOptions.ConcatenationOptions
    op.builtinOptions     = opts
    sg.operators.append(op)

    # ── replace model outputs ─────────────────────────────────────────────
    sg.outputs = [concat_out]

    # ── serialise ─────────────────────────────────────────────────────────
    ser = (getattr(fbu, 'convert_object_to_bytearray', None) or
           getattr(fbu, 'write_model', None))
    if ser is None:
        raise RuntimeError("flatbuffer_utils has no known serializer")

    if 'convert_object_to_bytearray' in ser.__name__:
        result = ser(model)
        OUT_MODEL.write_bytes(bytes(result))
    else:
        ser(model, str(OUT_MODEL))


# ── verify ────────────────────────────────────────────────────────────────────
def verify():
    import tensorflow as tf
    interp = tf.lite.Interpreter(str(OUT_MODEL))
    interp.allocate_tensors()
    print("\nVerification – output tensors of new model:")
    for d in interp.get_output_details():
        print(f"  name='{d['name']}'  shape={list(d['shape'])}  dtype={d['dtype'].__name__}")
    inp = interp.get_input_details()[0]
    print(f"\nInput:  shape={list(inp['shape'])}  dtype={inp['dtype'].__name__}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Input:  {IN_MODEL}")
    print(f"Output: {OUT_MODEL}\n")

    sfb, fbu = _probe()
    if sfb is None:
        print("ERROR: no mutable TFLite schema API found in this TF installation.")
        print("Expected tensorflow.lite.python.schema_py_generated (TF ≥ 2.x).")
        sys.exit(1)

    build_model(sfb, fbu)
    print(f"Written: {OUT_MODEL}  ({OUT_MODEL.stat().st_size:,} bytes)")
    verify()


if __name__ == '__main__':
    main()
