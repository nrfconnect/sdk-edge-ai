# Axon accelerated FOMO demo application

## Requirements

* Setting VDDIO of DK to 3.3V - can be done by Board Configurator in nRF Connect for Desktop
* Arducam Mega SPI Camera - tested with Arducam Mega 5MP B0401
* Connecting the camera SPI interface to selected GPIOs:
  * SCK - port 1, pin 4
  * MISO - port 1, pin 5
  * MOSI - port 1, pin 6
  * CS - port 1, pin 7

## How to build and use

Apply patches for other modules using `west patch apply`.
Then build for `nrf54lm20dk/nrf54lm20a/cpuapp` target and flash the DK.

Every 0.5s camera will capture image and application will run preprocessing, inference and postprocessing of results.
Results will be printed on second UART console (default).
Model should detect beer bottles and soda cans.
