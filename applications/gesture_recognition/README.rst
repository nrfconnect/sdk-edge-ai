.. _app_gesture_recognition:

Gesture recognition
###################

.. contents::
   :local:
   :depth: 2

Overview
********

.. note::

   Currently, the gesture recognition application supports only Neuton CPU-based neural network models.

This project demonstrates a gesture-based remote control device using Nordic Edge AI Lab solution.
The development kit can connect to a PC via Bluetooth Low Energy (BLE) as a HID device.
With gestures, a user can control media playback or presentation slides.
Based on accelerometer and gyroscope data, the nRF Edge AI model recognizes eight gesture classes: Swipe Right, Swipe Left, Double Shake, Double Tap, Rotation Clockwise, Rotation Counter-Clockwise, No Gestures (IDLE), and Unknown Gesture.
The neural network model is trained using `Nordic Edge AI Lab platform`_.
The whole process how to capture data and train the model is described in the `Nordic Edge AI Lab documentation`_.
A use-case demonstration video is available at `Gesture recognition use-case demo video`_.

Requirements
************

The application supports the following development kits:

.. table-from-sample-yaml::

Hardware used
*************

Development kits used
=====================

.. tabs::

   .. tab:: Thingy53

      `Nordic Thingy:53 Multi-protocol IoT prototyping platform`_

      The Nordic Thingy:53 is an easy-to-use IoT prototyping platform.
      It makes it possible to create prototypes and proofs-of-concept without building custom hardware.
      The Thingy:53 is built around the nRF5340 SoC, a dual-core wireless SoC.
      The processing power and memory size of its dual Arm Cortex-M33 processors enable it to run embedded machine learning (ML) models directly on the device.

      The Thingy:53 also includes multiple integrated sensors, such as environmental sensors, color and light sensors, accelerometers, and a magnetometer.
      It is powered by a rechargeable Li-Po battery that can be charged via USB-C.
      There is also an external 4-pin JST connector compatible with the Stemma/Qwiic/Grove standards for hardware accessories.

      .. figure:: images/nordic_thingy.jpg
         :alt: Nordic Thingy:53 kit

   .. tab:: nRF54LM20A

      `nRF54LM20A`_

      Requires a sensor Evaluation Board:

      .. list-table::
         :header-rows: 0

         * - .. figure:: images/sensor_EB_top.jpeg
                :alt: Sensor Evaluation Board top view
                :scale: 50%
           - .. figure:: images/sensor_EB_bottom.jpeg
                :alt: Sensor Evaluation Board bottom view
                :scale: 50%

      The Sensor Evaluation Board is connected to the DK via the "EXP" port on the nRF54LM20A board:

      .. figure:: images/nrf54lm20a_EB.jpeg
         :alt: nRF54LM20A sensor EXP port
         :scale: 50%

   .. tab:: nRF54L15 TAG

      `nRF54L15TAG`_

      The nRF54L15 TAG is a development board for the nRF54L15 chip.
      It is a small, low-cost development board that is perfect for prototyping and testing.

      .. list-table::
         :header-rows: 0

         * - .. figure:: images/nrf54l15tag_top.jpeg
                :alt: nRF54L15 TAG top view
                :scale: 50%
           - .. figure:: images/nrf54l15tag_bottom.jpeg
                :alt: nRF54L15 TAG bottom view
                :scale: 50%

Sensor BMI270
=============

On all development kits, the BMI270 sensor is used to collect data used to recognize gestures.
The `Bosch BMI270`_ is a 3-axis accelerometer and 3-axis gyroscope IMU sensor.


Setup software environment
**************************

.. TODO: Update when setting environment is ready; remove most of the content from this section.

For full instructions on preparing your development environment, see :ref:`setting_up_environment`.



User interface
**************

Button
======

The project has two keyboard control modes: Presentation Control and Music Control.
Depending on the control mode, recognized gestures are mapped to different keyboard keys.
Switch between control modes by pushing the user button:

.. tabs::

   .. tab:: Thingy53

      On Thingy53, press the button on top of the device to switch between Presentation Control and Music Control modes.

   .. tab:: nRF54LM20A

      On nRF54LM20A, press the user button (labeled ``BUTTON 0``) to switch between Presentation Control and Music Control modes.

   .. tab:: nRF54L15 TAG

      On nRF54L15 TAG, press the user button (labeled ``BTN1``) to switch between Presentation Control and Music Control modes.

LEDs
====

LED colour indicates the control mode:

.. tabs::

   .. tab:: Thingy53

      If the device is not connected to Bluetooth, the LED glows red.
      If the device is in Presentation Control mode, the LED glows blue.
      If the device is in Music Control mode, the LED glows green.
   
      .. list-table:: LED indication in different device states
         :header-rows: 1

         * - No Bluetooth connection
           - Presentation Control mode
           - Music Control mode
         * - .. figure:: images/device-led-no-ble-connect_thingy.gif
               :alt: No BLE connection
               :scale: 50%
           - .. figure:: images/device-led-ble-connect-presentation-mode_thingy.gif
               :alt: Presentation mode
               :scale: 50%
           - .. figure:: images/device-led-ble-connect-music-mode_thingy.gif
               :alt: Music mode
               :scale: 50%

   .. tab:: nRF54LM20A

      If the device is not connected to Bluetooth, the LED0 glows.
      If the device is in Presentation Control mode, the LED2 glows.
      If the device is in Music Control mode, the LED1 glows.

      .. list-table:: LED indication in different device states
         :header-rows: 1

         * - No Bluetooth connection
           - Presentation Control mode
           - Music Control mode
         * - .. figure:: images/device-led-no-ble-connect_nrf54lm20.gif
               :alt: No BLE connection
               :scale: 50%
           - .. figure:: images/device-led-ble-connect-presentation-mode_nrf54lm20.gif
               :alt: Presentation mode
               :scale: 50%
           - .. figure:: images/device-led-ble-connect-music-mode_nrf54lm20.gif
               :alt: Music mode
               :scale: 50%

   .. tab:: nRF54L15 TAG

      If the device is not connected to Bluetooth, the LED glows red.
      If the device is in Presentation Control mode, the LED glows blue. 
      If the device is in Music Control mode, the LED glows green.

      .. list-table:: LED indication in different device states
         :header-rows: 1

         * - No Bluetooth connection
           - Presentation Control mode
           - Music Control mode
         * - .. figure:: images/device-led-no-ble-connect_nrf54l15tag.jpeg
               :alt: No BLE connection
               :scale: 50%
           - .. figure:: images/device-led-ble-connect-presentation-mode_nrf54l15tag.jpeg
               :alt: Presentation mode
               :scale: 50%
           - .. figure:: images/device-led-ble-connect-music-mode_nrf54l15tag.jpeg
               :alt: Music mode
               :scale: 50%


Configuration
*************

|config|


Data collection firmware build
==============================

It is possible to create a special build that outputs raw data from the accelerometer and gyro sensors on the serial port. This makes it possible to capture data for training new models and to test and implement new use cases. The output consists of 16-bit integers separated by a comma, in the following order:

.. code-block::

   <acc_x>,<acc_y>,<acc_z>,<gyro_x>,<gyro_y>,<gyro_z>

Column headers are not included. The output rate is the configured sampling frequency (default 100 Hz).

To build this version, enable the following option in the ``prj.conf`` file:

.. code-block::

   CONFIG_DATA_COLLECTION_MODE=y

To forward the same data over BLE using `Nordic UART Service (NUS)`_, also enable:

.. code-block::

   CONFIG_DATA_COLLECTION_BLE_NUS=y

This mode requires an additional development kit running the `Nordic central UART sample`_ to receive the NUS data.

No inference is performed in this mode.
It is intended to simplify the capture of new datasets.


A raw dataset used for model training is available at `training dataset`_.


Building and running
********************

.. |sample path| replace:: :file:`applications/gesture_recognition`

Building
========

For command-line builds from the application directory, use the following board identifiers:

.. tabs::

   .. tab:: Thingy53

      .. code-block:: console

         west build -p -b thingy53/nrf5340/cpuapp
         west flash

   .. tab:: nRF54LM20A

      .. code-block:: console

         west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp
         west flash

   .. tab:: nRF54L15 TAG

      Insert nRF54L15 TAG board into nrf54l15dk debug out header and power on nrf54l15dk.

      .. code-block:: console

         west build -p -b nrf54l15tag/nrf54l15/cpuapp
         west flash


For more details on building the application, see `Building an application`_.

Testing
=======

|test_application|

1. |connect_kit|
#. |connect_terminal_kit|
   Connect to the serial device printing console output.
   It can be identified by output similar to the following:

   .. parsed-literal::
      :class: highlight

      \*\*\* Booting nRF Connect SDK v3.2.0-5dcc6bd39b0f \*\*\*
      \*\*\* Using Zephyr OS v4.2.99-a57ad913cf4e \*\*\*

#. When performing gestures with the device, the serial port terminal displays messages similar to the following:

   .. parsed-literal::
      :class: highlight

      Predicted class: DOUBLE SHAKE, with probability 96 %
      BLE HID Key 8 sent successfully
      Predicted class: SWIPE RIGHT, with probability 99 %
      BLE HID Key 32 sent successfully
      Predicted class: SWIPE LEFT, with probability 99 %
      BLE HID Key 16 sent successfully
      Predicted class: ROTATION RIGHT, with probability 93 %
      BLE HID Key 1 sent successfully

   Once the device is running, BLE advertising starts as a HID device and waits for a connection request from the PC.
   Devices can be connected in the same way as a regular Bluetooth keyboard.

#. Adding the Bluetooth device varies depending on the operating system.
   For Windows 10, go to **Settings** > **Bluetooth & other devices** > **Add Bluetooth or other device**.

   .. figure:: images/ble_connect_1.png
      :alt: Add Bluetooth device

#. The device appears in the **Add a device** window. Select the device for pairing.

   .. figure:: images/device_ble_scanning.jpg
      :alt: BLE device scanning

#. After pairing, the device appears in the **Mouse, keyboard, & pen** section.

   .. figure:: images/device_ble_connected.jpg
      :alt: BLE device connected

#. In the serial port terminal, the following log messages appear:

   .. code-block::

      Connected 9C:B6:D0:C0:CE:FC (public)
      Security changed: 9C:B6:D0:C0:CE:FC (public) level 2
      Input CCCD enabled
      Input attribute handle: 0
      Consumer CCCD enabled

   After Bluetooth connection, the device changes LED indication from red to green or blue depending on the keyboard control mode.

#. You can now use the device to control media playback or presentation slides by making gestures.

Gestures
========

.. list-table:: Gestures to keyboard keys mapping
   :header-rows: 1

   * - Gesture
     - Presentation Control - blue LED
     - Music Control - green LED
   * - Double Shake
     - F5
     - Media Play/Pause
   * - Double Tap
     - Escape
     - Media Mute
   * - Swipe Right
     - Arrow Right
     - Media Next
   * - Swipe Left
     - Arrow Left
     - Media Previous
   * - Rotation Clockwise
     - Not used
     - Media Volume Up
   * - Rotation Counter-Clockwise
     - Not used
     - Media Volume Down

How to make gestures
====================

.. note::
   The model was trained with a limited dataset.
   For optimal gesture recognition across different users, follow the instructions carefully and train a new model with a larger dataset.

Make sure the default (initial) position of the device matches the following:

.. tabs::

   .. tab:: Thingy53

      .. figure:: images/initial_orientation_thingy.gif
         :alt: Initial orientation (Thingy53)

   .. tab:: nRF54LM20A

      .. figure:: images/initial_orientation_54lm20.gif
         :alt: Initial orientation (nRF54LM20A)

   .. tab:: nRF54L15 TAG

      .. figure:: images/initial_orientation_nrf54l15tag.gif
         :alt: Initial orientation (nRF54L15 TAG)

Follow the images below to make gestures.
For better recognition, use your wrists more than your whole hand.
The gestures are performed with the device in the initial position.
Keeping the device orientation is important for the recognition to work correctly.

.. tabs::

   .. tab:: Thingy53

      .. list-table:: Swipe right and left
         :header-rows: 1

         * - Swipe Right
           - Swipe Left
         * - .. figure:: images/swipe_right_thingy.gif
                :alt: Swipe right
           - .. figure:: images/swipe_left_thingy.gif
                :alt: Swipe left

      .. list-table:: Rotation clockwise and counter-clockwise
         :header-rows: 1

         * - Rotation Clockwise
           - Rotation Counter-Clockwise
         * - .. figure:: images/rotation_right_thingy.gif
                :alt: Rotation clockwise
           - .. figure:: images/rotation_left_thingy.gif
                :alt: Rotation counter-clockwise

      .. list-table:: Double shake and double tap
         :header-rows: 1

         * - Double Shake
           - Double Tap
         * - .. figure:: images/double_shake_thingy.gif
                :alt: Double shake
           - .. figure:: images/double_tap_thingy.gif
                :alt: Double tap

   .. tab:: nRF54LM20A

      .. list-table:: Swipe right and left
         :header-rows: 1

         * - Swipe Right
           - Swipe Left
         * - .. figure:: images/swipe_right_nrf54lm20.gif
                :alt: Swipe right
           - .. figure:: images/swipe_left_nrf54lm20.gif
                :alt: Swipe left

      .. list-table:: Rotation clockwise and counter-clockwise
         :header-rows: 1

         * - Rotation Clockwise
           - Rotation Counter-Clockwise
         * - .. figure:: images/rotation_right_nrf54lm20.gif
                :alt: Rotation clockwise
           - .. figure:: images/rotation_left_nrf54lm20.gif
                :alt: Rotation counter-clockwise

      .. list-table:: Double shake and double tap
         :header-rows: 1

         * - Double Shake
           - Double Tap
         * - .. figure:: images/double_shake_nrf54lm20.gif
                :alt: Double shake
           - .. figure:: images/double_tap_nrf54lm20.gif
                :alt: Double tap

   .. tab:: nRF54L15 TAG

      .. list-table:: Swipe right and left
         :header-rows: 1

         * - Swipe Right
           - Swipe Left
         * - .. figure:: images/swipe_right_nrf54l15tag.gif
                :alt: Swipe right
           - .. figure:: images/swipe_left_nrf54l15tag.gif
                :alt: Swipe left

      .. list-table:: Rotation clockwise and counter-clockwise
         :header-rows: 1

         * - Rotation Clockwise
           - Rotation Counter-Clockwise
         * - .. figure:: images/rotation_right_nrf54l15tag.gif
                :alt: Rotation clockwise
           - .. figure:: images/rotation_left_nrf54l15tag.gif
                :alt: Rotation counter-clockwise

      .. list-table:: Double shake and double tap
         :header-rows: 1

         * - Double Shake
           - Double Tap
         * - .. figure:: images/double_shake_nrf54l15tag.gif
                :alt: Double shake
           - .. figure:: images/double_tap_nrf54l15tag.gif
                :alt: Double tap

Have fun and use this model for your future gesture control projects.

Dependencies
************

This sample uses the following |NCS| services:

* `Nordic UART Service (NUS)`_

This sample uses the following Zephyr libraries:

* `Logging`_
* `Sensor`_
