# Visual wake word / MCUNet VWW (nRF54LM20 DK, Axon)

Runs **mcunet_vww_320kb** on **ArduCam Mega** frames at **128×128 RGB565**. The **144×144** model input is padded with neutral gray to match training (`symmetric_m1_1`).

**Board:** `nrf54lm20dk` (`boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay`).

## Camera on the DK

The nRF54LM20 DK has **no Arduino-style camera header**. Wire the ArduCam Mega **SPI** module:

| ArduCam Mega | nRF54LM20 DK |
| --- | --- |
| VCC | **3.3 V** on the kit (check your module’s voltage spec) |
| GND | **GND** |
| SCK | **P1.04** |
| MISO | **P1.05** |
| MOSI | **P1.06** |
| CS | **P1.07** |

SPI **spi22**, **8 MHz**, active-low CS — details in the overlay.

Pinout reference: [nRF54LM20 DK](https://docs.nordicsemi.com/bundle/ncs-latest/page/zephyr/boards/nordic/nrf54lm20dk/doc/index.html).

## Model and compiler

1. Put **`models/mcunet-320kb-1mb_vww.tflite`** in place (path from `compiler_mcunet_vww_320kb.yaml`).
2. From `sdk-edge-ai/tools/axon/compiler` (Docker):

```bash
./run_docker.sh axon_compiler \
  /absolute/path/to/sdk-edge-ai/applications/VWW/compiler_mcunet_vww_320kb.yaml
```

Output: **`outputs/nrf_axon_model_mcunet_vww_320kb_.h`** (and related files). On failure: unsupported ops → toolchain/model change; buffer errors → raise `interlayer_buffer_size` / `psum_buffer_size` in the YAML.

**RAM:** Keep `CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE` in `prj.conf` just above `MAX_IL_BUFFER_USED` from the generated header. Oversizing wastes SRAM and can break the link.

## Build and flash

```bash
west build -b nrf54lm20dk/nrf54lm20b/cpuapp sdk-edge-ai/applications/VWW
west flash
```

## Logs and class index

Firmware uses **`nrf_axon_nn_get_classification`**; class **1** is **person** per `compiler_mcunet_vww_320kb.yaml`. If your labels are swapped, fix interpretation in `main.c`.

`CONFIG_CBPRINTF_FP_SUPPORT=y` enables float logs for probabilities.

## When labels or scores look wrong

Use **`scripts/check_tflite_vs_axon.py`** so TFLite and Axon **H×W** match. Then **`scripts/run_tflite_reference.py`** on still images and, if needed, **`scripts/visualize_compare_inputs.py`** — see script `--help`. After changing the model, re-run the Axon compiler and a **pristine** `west build` so headers and firmware stay in sync.

## Simulator

`simulator/` — see `simulator/README.md`. Prefer TFLite on PC if the Axon sim rejects the graph.
