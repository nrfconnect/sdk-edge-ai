# Person recognition (nRF54L, Axon)

Runs **mcunet_vww_320kb** (MCUNet Visual Wake Word int8 TFLite compiled for Axon) on one or more embedded JPEG-derived inputs and logs **person present: yes/no** via `nrf_axon_nn_get_classification` (2 classes: non_person, person).

## 1) Add the TFLite model

Copy the file you mentioned into `models/` with the exact name the compiler YAML expects:

```bash
cp ~/.torch/mcunetmcunet-320kb-1mb_vww.tflite \
   sdk-edge-ai/applications/person_recognition/models/
```

See `models/README.md` for details.

## 2) Compile the model for Axon

From `sdk-edge-ai/tools/axon/compiler` (Docker must be available):

```bash
./run_docker.sh axon_compiler \
  /absolute/path/to/sdk-edge-ai/applications/person_recognition/compiler_mcunet_vww_320kb.yaml
```

The compiler uses the YAML’s directory as the workspace, so `models/…` and `outputs/` are under `applications/person_recognition/`.

**Artifacts:** `outputs/nrf_axon_model_mcunet_vww_320kb_.h` (and related files).

If compilation fails:

- Unsupported operator: the graph may need changes or a newer SDK/compiler.
- Buffer too small: increase `interlayer_buffer_size` / `psum_buffer_size` in `compiler_mcunet_vww_320kb.yaml` and retry.

### After `check_tflite_vs_axon.py` reports H×W mismatch

The board uses whatever is in `outputs/nrf_axon_model_*.h`. If that size ≠ your `.tflite`, PC and device are not the same graph input.

1. Ensure `models/mcunet-320kb-1mb_vww.tflite` is the file you want (the 144×144 one that matches `run_tflite_reference.py`).
2. Re-run the Axon compiler (Docker) on `compiler_mcunet_vww_320kb.yaml` so `outputs/` is regenerated from **that** file.
3. Run `python3 scripts/check_tflite_vs_axon.py ...` again — **TFLite H×W and Axon H×W must match.**
4. If the header is still wrong, toggle `reshape_input` in the YAML (`false` is set for mcunet; compiler default was `true`) and recompile.
5. Update `prj.conf` `CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE` if `MAX_IL_BUFFER_USED` in the new header changed, then **pristine** `west build` so `test_images.h` regenerates.

## 3) RAM (Axon interlayer buffer)

`CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE` allocates a **.bss** buffer in SRAM. Setting it far above the model’s real need (e.g. 800000 when the model needs ~147 KiB) will **overflow RAM** at link time.

`prj.conf` is tuned to **just above** `NRF_AXON_MODEL_MCUNET_VWW_320KB_MAX_IL_BUFFER_USED` from `outputs/nrf_axon_model_mcunet_vww_320kb_.h`. If you recompile the model and the static assert in that header fails, increase the interlayer size in small steps. Set `CONFIG_NRF_AXON_PSUM_BUFFER_SIZE` from `MAX_PSUM_BUFFER_USED` (often 0).

**Note:** Embedded JPEG tensors live in **flash** (`const`); using one picture saves **flash**, not SRAM, unless you also shrink other allocations.

## 4) Test pictures

In `pictures/` (at least `demo_picture.jpeg` for the default build):

- `demo_picture.jpeg` (always required for default embed)
- `demo_2.jpeg`, `demo_3.jpeg` (optional; enable below)

By default CMake embeds **only** `demo_picture.jpeg` to save flash. To embed all three:

```bash
west build -b <board> sdk-edge-ai/applications/person_recognition \
  -DPERSON_RECOGNITION_EMBED_ALL_PICTURES=ON
```

Or run the script manually, e.g. `--images demo_picture.jpeg`.

At build time, `scripts/embed_test_images.py` reads **input shape and Axon quant parameters** from `outputs/nrf_axon_model_mcunet_vww_320kb_.h`.

**Preprocessing must match `eval_det.py`:** MCUNet VWW TFLite expects floats derived as `(pixel/255)*2 - 1` (range **[-1, 1]**) before quantization, **not** `[0, 1]`. CMake passes `--preprocess symmetric_m1_1`. Using `[0, 1]` was a common bug and breaks accuracy badly.

## Troubleshooting (wrong labels on device)

1. **TFLite vs Axon spatial size** — The `.tflite` input `H×W` must match the compiled header. Check:

   ```bash
   cd sdk-edge-ai/applications/person_recognition
   python3 scripts/check_tflite_vs_axon.py \
     --tflite models/mcunet-320kb-1mb_vww.tflite \
     --axon-header outputs/nrf_axon_model_mcunet_vww_320kb_.h
   ```

   If this reports a mismatch (e.g. 144×144 in `.tflite` vs 146×146 in the header), **re-run the Axon compiler** on the **same** `.tflite` you intend to use, then rebuild the app.

2. **Visualize `test_images.h` vs PC** — Decode the embedded int8 tensors back to RGB and place them next to the TFLite (144×144) pipeline:

   ```bash
   python3 scripts/visualize_compare_inputs.py --out-dir debug_inputs
   ```

   Outputs per JPEG: `*_device_from_header.png` (what the MCU sees after dequant), `*_pc_tflite_144x144.png`, `*_side_by_side.png`, plus `debug_inputs/compare_report.txt`. If “parsed header matches embed regeneration” appears, `test_images.h` is consistent with `embed_test_images.py`.

3. **PC reference for the same JPEGs** — Compare the board to TFLite on your PC:

   ```bash
   python3 scripts/run_tflite_reference.py \
     --tflite models/mcunet-320kb-1mb_vww.tflite \
     pictures/demo_picture.jpeg pictures/demo_2.jpeg pictures/demo_3.jpeg
   ```

   If PC results are already wrong, the issue is data/labels/model, not the firmware. If PC is right but the board wrong, compare `H×W`, preprocessing, and that you flashed a build that regenerated `test_images.h` after fixing embed.

4. **Class order** — Firmware assumes index **1** is “person” (as in `compiler_mcunet_vww_320kb.yaml` labels). If your training swapped classes, invert the interpretation in `main.c`.

## Axon host simulator (optional)

A small CMake target that links the same model header against the PC Axon simulator lives in `simulator/`. It **builds** on Linux when Nordic’s prebuilt `libnrf-axon-*` libraries are present; **running** `mcunet_vww_320kb` may hit simulator limitations (see `simulator/README.md`). Use TFLite on PC for MCUNet reference output if the sim rejects your graph.

## 5) Build and flash

```bash
west build -b <your_nrf54l_board> sdk-edge-ai/applications/person_recognition
west flash
```

## Output

For each image, the firmware logs:

- **person present** from `nrf_axon_nn_get_classification` (argmax on packed int32 logits).
- **P(person)** and **dequantized logits** using the same linear dequantization as the Axon model header (`output_dequant_*`) and a 2-class softmax, aligned with `eval_det.py` (`_dequantize_output` + softmax on two outputs).

With identical scale/zero-point for both logits, argmax on quantized values matches argmax on floats; probability still needs float logits and `expf`.

`CONFIG_CBPRINTF_FP_SUPPORT=y` is set so `LOG_INF` can print probabilities.

## Switching presets without Axon header

For development without a compiled header yet, you can run the embed script manually with a preset:

```bash
python3 scripts/embed_test_images.py --model vww --out src/generated/test_images.h
```

Firmware still requires `outputs/nrf_axon_model_mcunet_vww_320kb_.h` for CMake.
