# mcunet_vww_320kb — Axon host simulator

Runs the compiled `nrf_axon_model_mcunet_vww_320kb_.h` on the **Axon PC simulator** (same inference path as nRF54L, without Zephyr).

## Prerequisites

- Prebuilt Axon libraries under `sdk-edge-ai/lib/axon/bin/Linux/` (or Darwin/Windows) and `lib/axon/platform/bin/simulator/<OS>/` (shipped with the Edge AI add-on).
- CMake ≥ 3.13, a C compiler, and **pthread** on Linux.

## Build

From this directory:

```bash
cmake -B build -S .
cmake --build build
```

## Run

**Zeros only** (sanity check):

```bash
./build/mcunet_vww_axon_sim
```

**Same int8 tensor as firmware** — export a raw CHW buffer from a JPEG, then:

```bash
python3 ../scripts/export_mcunet_input_bin.py --jpeg ../pictures/demo_picture.jpeg --out /tmp/in.bin
./build/mcunet_vww_axon_sim /tmp/in.bin
```

Compare the printed **classification idx / score** and raw output bytes to UART logs from the board.

## Simulator vs this model

The PC simulator **does not support every** compiled graph. With some MCUNet builds you may see:

```text
ERROR: check_cfg: Pointwise convolution   4 <= cfg_in_fmap_size_x. cfg_in_fmap_size_x=0x1
axon-nn simulator failure -1001
```

That comes from the closed-source Axon NN simulator library, not from this sample `main`. The same `nrf_axon_model_mcunet_vww_320kb_.h` can still run on **silicon**. To confirm the simulator toolchain on your machine, build `tests/axon/inference/simulator` with `-DNRF_AXON_MODEL_NAME=tinyml_vww` — that run should pass.

If you need host-side golden checks for MCUNet, use **TFLite on PC** (`scripts/run_tflite_reference.py`) until Nordic extends simulator coverage or you use a graph variant the simulator accepts.

## Model header path

The build includes `../outputs/nrf_axon_model_mcunet_vww_320kb_.h`. Re-run the Axon compiler after changing the model.

## Official generic harness

The shared test harness (multiple sample models + golden vectors) lives under `sdk-edge-ai/tests/axon/inference/simulator/` and expects `nrf_axon_model_<name>_test_vectors_.h`. This app avoids that by feeding **your** binary input.
