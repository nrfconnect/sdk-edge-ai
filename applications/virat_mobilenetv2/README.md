# Compile virat_mobilenetv2.keras for Axon

You can either run in **one step** (Keras → TFLite → Axon inside Docker) or **two steps** (convert .keras → .tflite locally, then compile the .tflite for Axon).

---

## Two-step: .keras → .tflite, then compile

### Step 1: Convert .keras to .tflite

Place **models/virat_mobilenetv2.keras** and **data/x_train_virat.npy**. Generate placeholder data if needed:

```bash
cd sdk-edge-ai/applications/virat_mobilenetv2
python3 scripts/generate_train_data.py --model models/virat_mobilenetv2.keras --out data/x_train_virat.npy
```

Then convert to int8 TFLite:

```bash
python3 scripts/keras_to_tflite.py --keras models/virat_mobilenetv2.keras --train-data data/x_train_virat.npy --out models/virat_mobilenetv2.tflite
```

This writes **models/virat_mobilenetv2.tflite**.

### Step 2: Compile .tflite for Axon

From **sdk-edge-ai/tools/axon/compiler** (with Docker):

```bash
./run_docker.sh axon_compiler /path/to/sdk-edge-ai/applications/virat_mobilenetv2/compiler_virat_mobilenetv2_tflite.yaml
```

Outputs go to **applications/virat_mobilenetv2/outputs/** (e.g. `nrf_axon_model_virat_mobilenetv2_.h`).

---

## One-step: Keras → Axon in one compiler run

Place **models/virat_mobilenetv2.keras** and **data/x_train_virat.npy**. The compiler converts to TFLite internally, then compiles. Run:

```bash
./run_docker.sh axon_compiler /path/to/sdk-edge-ai/applications/virat_mobilenetv2/compiler_virat_mobilenetv2.yaml
```

Outputs are in **outputs/**.

---

## Place files (for either flow)

- **models/virat_mobilenetv2.keras** — your Keras model (`.h5` is also supported).
- **data/x_train_virat.npy** — representative training data: NumPy array shape `(N, H, W, C)` matching the model input. The compiler uses ~10% for quantization. You can generate a placeholder from the model’s input shape:

  ```bash
  python3 scripts/generate_train_data.py --model models/virat_mobilenetv2.keras --out data/x_train_virat.npy
  ```

## Outputs: .h vs .npy

- **.h files** (in **outputs/**) — these are what you use in firmware: `nrf_axon_model_<name>_.h`, and optionally test-vector headers. The C compiler library step inside the Docker run produces them. Your app should include from **outputs/**.
- **.npy files** (in **intermediate/**) — intermediate only. If you have `get_quantized_data: true` in the YAML, the compiler also writes e.g. `<model_name>_q_data.npy` (quantized test data). That is **not** a replacement for the .h files.

If you only see .npy and no .h files:

1. Check **outputs/** in the same workspace (not only **intermediate/**).
2. Check the compiler log for errors: the step that runs the C compiler library may have failed (e.g. unsupported ops, or missing compiler binary in the image).
3. You cannot use the .npy alone to run the model on the device; you need the generated .h files. The .npy can be used as **test_data** in a later compiler run (so test vectors get embedded in the .h), or converted to a C array for embedding (see script below).

### Optional: convert .npy to a C header (for one test vector)

If you have a quantized input array in .npy (e.g. one sample, shape `(1, H, W, C)` or `(H, W, C)` int8) and want to embed it as a C array:

```bash
python3 scripts/npy_to_c_header.py --npy intermediate/virat_mobilenetv2_q_data.npy --slice 0 --out src/input_vector.h
```

This writes a single int8 array to the header. Use it only if you already have the **model** .h from a successful full compile and just need an input vector in C.
