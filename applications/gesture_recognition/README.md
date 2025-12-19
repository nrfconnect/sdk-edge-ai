# nRF Edge AI Nordic Thingy:53 Gesture Based BLE Remote Control Device

- [Overview](#overview)
- [Hardware Used](#hw-used)
- [Setup Software Environment](#setup-sw-env)
- [Setup Firmware Project](#setup-fw-proj)
- [How The Project Works](#how-works)

## Overview <div id='overview'/>

This project demonstrates a gesture based remote control device using [__Nordic Thingy:53__](https://www.nordicsemi.com/Products/Development-hardware/Nordic-Thingy-53). The development kit could be connected to the PC via Bluetooth as a HID device and using gestures the user can control media stream or slides of the presentation. Based on accelerometer and gyroscope data the nRF Edge AI model could recognize __8 classes__ of gestures: Swipe Right, Swipe Left, Double Shake, Double Tap, Rotation Clockwise and Counter clockwise, No Gestures(IDLE) and Unknown Gesture. Use-case demonstration [video](https://youtu.be/qDFdxapLbrA). Raw dataset used for model training, which you can use to train your own model, or augment it with your own data and train a more robust model is located [here](https://files.nordicsemi.com/artifactory/edge-ai/external/nordic53thingy_remote_ctrl_train_v101.csv).

## Hardware Used <div id='hw-used'/>

[__Nordic Thingy:53 Multi-protocol IoT prototyping platform__](https://www.nordicsemi.com/Products/Development-hardware/Nordic-Thingy-53)

The Nordic Thingy:53™ is an easy-to-use IoT prototyping platform. It makes it possible to create prototypes and proofs-of-concept without building custom hardware. The Thingy:53 is built around the nRF5340 SoC, the flagship dual-core wireless SoC. The processing power and memory size of its dual Arm Cortex-M33 processors enables it to run embedded machine learning (ML) models directly on the device.

The Thingy:53 also includes many different integrated sensors, like environmental-, and color and light sensors, accelerometers, and a magnetometer, which all can be taken advantage of without additional hardware. It is powered by a rechargeable Li-Po battery that can be charged via USB-C. There is also an external 4-pin JST connector compatible with the Stemma/Qwiic/Grove standards for hardware accessories.

![nordic-thingy-kit-img](resources/nordic_thingy.jpg)

## Setup Software Environment <div id='setup-sw-env'/>

To set this project up, you will need to install the following software:
- Visual Studio Code (https://code.visualstudio.com)
- nRF Connect for VS Code (https://www.nordicsemi.com/Products/Development-tools/nRF-Connect-for-VS-Code)
- [**Optional**] TeraTerm Terminal (https://teratermproject.github.io/index-en.html)

## Setup Firmware Project <div id='setup-fw-proj'/>

1. Clone this repository: https://github.com/Neuton-tinyML/neuton-nordic-thingy53-ble-remotecontrol
2. In the VS code, `Open folder` where you did clone the repository.
3. `Manage toolchain` install toolchain version **v3.1.0**. `Manage SDK` install version **v3.1.0** in nRF Connect tab.

![sw-install-step3-img](resources/sw-install-step-3.jpg)

4. `Add Build configuration` for Nordic Thingy 53 dev kit

![sw-install-step4-img](resources/sw-install-step-4.jpg)

5. Choose `thingy53_nrf5340_cpuapp` in the `Board` selector and click `Build Configuration`

![sw-install-step5-img](resources/sw-install-step-5.jpg)

6. After your build will be configured you should see the following options:

![sw-install-step6-img](resources/sw-install-step-6.jpg)

**IMPORTANT** If your `thingy53_nrf5340_cpuapp_defconfig` file does not has **`CONFIG_FPU=y`** you should add this because nRF Edge AI library is compiled with `-mfloat-abi=hard` flag

![sw-install-step5-1-img](resources/sw-install-step-5-1-important.jpg)

8. Now turn on your Thingy 53 dev kit and connect to your PC via debugger and USB

![sw-install-step7-img](resources/connected-devkit-sw-install-step-7.jpg)

9. Build & Flash the device with firmware

![sw-install-step8-img](resources/sw-install-step-8.jpg)

10. After successfull device programming you can open any serial port terminal and you should see the following messages:

````
*** Booting nRF Connect SDK 7a22da43c1d4 ***
Set up button at gpio@842800 pin 14
nRF Edge AI Gestures Recognition Demo:
         Application version: 3.0.0
         nRF Edge AI Runtime Version: 1.0.0
         nRF Edge AI Lab Solution id: 84622
Bluetooth initialized
Advertising successfully started
````

MacOs CLI commands examples:
- Check connected usb devices: `ls -l /dev/cu.usb*`
- Output to serial port. Use your actual usb-device name: `stty -f /dev/cu.usbmodem101 115200 | cat /dev/cu.usbmodem101`
- Save serial port output to file, if necessary: `stty -f /dev/cu.usbmodem101 115200 | cat /dev/cu.usbmodem101 | tee filename.csv`


11. Explore the project and nRF Edge AI model capabilities!

### Data collection firmware build

It is possible to create a special build that will output the raw data of the acceleromater and gyro sensors in the serial port.
This way it's possible to capture data for training new models, to test and implement new use cases.
The output of the sensors will integers of 16 bits, separated by a comma, in the following order

```
<acc_x>,<acc_y>,<acc_z>,<gyro_x>,<gyro_y>,<giro_z>
```
*(Please note column headers are not included)*

The output rate will be the configured sampling frequency, default value being 100Hz.

To build this version, the following option must be enabled in the `prj.conf` file

```
CONFIG_DATA_COLLECTION_MODE=y
```

The project must be build and flashed again as described in the step **(9)**.

No inference will be performed in this mode, it's just intended to simplify the capture of new datasets

# How the project works <div id='how-works'/>

Once the device is up and running, Bluetooth advertising starts as a HID device and waits for connection request from the PC.

You can connect devices in the same way as, for example, a regular Bluetooth keyboard.

1. For Windows 10 PC you can go to `Settings -> Bluetooth & other devices -> Add Bluetooth or other device`.

![bt_step_1](resources/ble_connect_1.png)

2. The device should appear in `Add a device` window, choose the device for pairing.

![bt_step_2](resources/device_ble_scanning.jpg)

3. After Bluetooth pairing the device should appear in your `Mouse, keyboard, & pen` section

![bt_step_2](resources/device_ble_connected.jpg)

4. In the serial port terminal you should se the following logs messages:

```
Connected 9C:B6:D0:C0:CE:FC (public)
Security changed: 9C:B6:D0:C0:CE:FC (public) level 2
Input CCCD enabled
Input attribute handle: 0
Consumer CCCD enabled
```

After Bluetooth connection the device will change LED indication from RED LED glowing to GREEN or BLUE LEDs glowing depending on __Keyboard control mode__. 

The project has two Keyboard control modes: __Presentation Control__ and __Music Control__. You can switch between control modes by pushing user button `BTN0`, for different control modes there is a different LED indication. If the device in __Presentation Control__ mode the LED glows BLUE color, if the device in __Music Control__ mode the LED glows GREEN color:

__LED indication in different device states:__

| No Bluetooth connection |  Presentation Control mode   |   Music Control mode    |
| ------------------------ |---------------------------- | ----------------------- |
| ![Alt Text](resources/device-led-no-ble-connect.gif) |![Alt Text](resources/device-led-ble-connect-presentation-mode.gif)      |![Alt Text](resources/device-led-ble-connect-music-mode.gif)  |


Depending on the control mode, recognized gestures will be mapped to different keyboard keys:

__Gestures to Keyboard Keys Mapping__

|        |   Presentation Control            | Music Control     |
| ----------------------------- | ---------- | ----------------- |
| Double Shake                  | F5         | Media Play/Pause  |
| Double Tap                    | ESCAPE     | Media Mute        | 
| Swipe Right                   | Arrow Right| Media Next        | 
| Swipe Left                    | Arrow Left | Media Previous    | 
| Rotation Clockwise            | Not used   | Media Volume Up   | 
| Rotation Counter clockwise    | Not used   | Media Volume Down |

__How to Make Gestures__
> **_NOTE:_**  The dataset for creating this model is immature and this affects the generalization of the model on different persons, so please follow the instructions for good gesture recognition.

To begin with, please make sure that the default (initial) position of the device is the same as following:

![gestures-img-1](resources/initial_orientation.gif)

Next, follow the images on how to make gestures. For better recognition use your wrists more when making gestures, and not your whole hand:

__Swipe Right & Left__
|                                        |                                               |
| -------------------------------------- | --------------------------------------------- |
| ![Alt Text](resources/swipe_right.gif) | ![Alt Text](resources/swipe_left.gif)        |
| Swipe Right                            |    Swipe Left                                |

__Rotation Clockwise & Counter Clockwise__

|                                           |                                               |
| --------------------------------------    | --------------------------------------------- |
| ![Alt Text](resources/rotation_right.gif) | ![Alt Text](resources/rotation_left.gif)      |
| Rotation Clockwise(Right)                 |    Rotation Counter Clockwise(Left)            |

__Double Shake & Double Tap__

|                                           |                                               |
| --------------------------------------    | --------------------------------------------- |
| ![Alt Text](resources/double_shake.gif)   | ![Alt Text](resources/double_tap.gif)      |
| Double Shake                              |    Double Tap                              |

When performing gestures with the device, in the serial port terminal, you should see the similar messages:

```
Predicted class: DOUBLE SHAKE, with probability 96 %
BLE HID Key 8 sent successfully
Predicted class: SWIPE RIGHT, with probability 99 %
BLE HID Key 32 sent successfully
Predicted class: SWIPE LEFT, with probability 99 %
BLE HID Key 16 sent successfully
Predicted class: ROTATION RIGHT, with probability 93 %
BLE HID Key 1 sent successfully
```
Have fun and use this model for your future gesture control projects!