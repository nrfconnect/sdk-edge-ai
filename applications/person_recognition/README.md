# Person detection (nRF54LM20 DK, Axon)

Runs the **person_det** Axon model on frames from an **ArduCam Mega** at **128×128 RGB565**. The 160×128 model input is padded horizontally with neutral gray (`main.c`).

**Board:** `nrf54lm20dk` (overlay in `boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay`).

## Camera on the DK

The nRF54LM20 DK has **no Arduino-style camera header**. Wire the ArduCam Mega **SPI** module to the DK (3.3 V logic).

| ArduCam Mega | nRF54LM20 DK |
| --- | --- |
| VCC | **VDD** / 3.3 V (use the kit’s 3.3 V, not 5 V unless your module requires 5 V only) |
| GND | **GND** |
| SCK | **P1.04** |
| MISO | **P1.05** |
| MOSI | **P1.06** |
| CS | **P1.07** |

SPI instance **spi22**, **8 MHz**, active-low CS — see the overlay if you remap pins.

Full pinout: [nRF54LM20 DK](https://docs.nordicsemi.com/bundle/ncs-latest/page/zephyr/boards/nordic/nrf54lm20dk/doc/index.html).

## Model and compiler

1. Place **`models/person-det.tflite`** (name must match `compiler_person_det_input.yaml`).
2. Add at least one image under **`pictures/`** (or pass `--image`). Then:  
   `python3 scripts/prepare_demo_input.py` → **`data/demo_input.npy`** (requires `models/person-det.tflite` and Pillow).
3. From `sdk-edge-ai/tools/axon/compiler` (Docker):

```bash
./run_docker.sh axon_compiler \
  /absolute/path/to/sdk-edge-ai/applications/person_recognition/compiler_person_det_input.yaml
```

Artifacts go under **`outputs/`**, including `nrf_axon_model_person_det_.h`. If compile fails, adjust `interlayer_buffer_size` / `psum_buffer_size` in the YAML and retry.

**RAM:** `CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE` in `prj.conf` must be ≥ `MAX_IL_BUFFER_USED` from that header (see static assert). After recompiling the model, bump the Kconfig value if needed and rebuild.

## Build and flash

```bash
west build -b nrf54lm20dk/nrf54lm20b/cpuapp sdk-edge-ai/applications/person_recognition
west flash
```

Open a UART console for logs.

## Runtime behavior

Logs detection results from the person-det head; **LED0** toggles while capturing, **LED1** indicates person (see `main.c`). For RGB565 dump/debug over UART, see `scripts/uart_rx_rgb565_frame.py`.

## Optional: PC checks

If results look wrong, align PC and device: same **input size** and **preprocessing** (symmetric \[-1, 1\] before quant). Scripts in `scripts/` (`check_tflite_vs_axon.py`, etc.) apply mainly when you have a matching TFLite and Axon header to compare.

## Simulator

`simulator/` can link the model for host builds; see `simulator/README.md`.
