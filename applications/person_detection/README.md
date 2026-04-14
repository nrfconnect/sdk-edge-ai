# Person detection (nRF54LM20 DK, Axon)

Runs the `person_det` Axon model on frames from an ArduCam Mega at 128×128 RGB565. The 160×128 model input is padded horizontally with neutral gray (`main.c`).

**Board:** `nrf54lm20dk` (overlay in `boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay`).

## Camera on the DK

**You need to use Board Configuratior to set VDD to 3.3V.**

Wire the ArduCam Mega SPI module to the DK (3.3 V logic).

| ArduCam Mega | nRF54LM20 DK |
| --- | --- |
| VCC | **VDDIO** |
| GND | **GND** |
| SCK | **P1.04** |
| MISO | **P1.05** |
| MOSI | **P1.06** |
| CS | **P1.07** |

## Runtime behavior

Logs detection results from the person-det head; **LED0** toggles while capturing, **LED1** indicates person detected. Captured image and model prediction is dumped over USB CDC ACM.

Some pins can be use tu trace key steps:
* **PIN1.10** - capturing image from camera
* **PIN1.11** - requantization of image
* **PIN1.12** - inference
* **PIN1.13** - postprocessing

## Result visualization

Run python script `scripts/live_usb_person_detection.py` with serial port of USB CDC ACM as first argument.

If the board is the only serial device connected to host than:
* `COM0`/`ttyACM0` and `COM1`/`ttyACM1` are for debugger
* `COM2`/`ttyACM2` is USB CDC ACM
