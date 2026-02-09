.. _app_gesture_recognition:

Gesture recognition
###################

.. contents::
   :local:
   :depth: 2

This application demonstrates a gesture-based remote control device using Nordic Edge AI Lab solution.

Application overview
********************

.. note::

   Currently, the gesture recognition application supports only Neuton CPU-based neural network models.

The gesture recognition application demonstrates how to use an nRF Edge AI model to recognize hand gestures from motion sensor data and expose them as standard HID inputs over Bluetooth® Low Energy.
When connected to a PC, the device appears as a Bluetooth LE HID device, allowing recognized gestures to control media playback or presentation slides.
Based on accelerometer and gyroscope data, the nRF Edge AI model recognizes eight gesture classes:

* Swipe right
* Swipe left
* Double shake
* Double tap
* Rotation clockwise
* Rotation counter-clockwise
* No gestures (IDLE)
* Unknown gesture

The neural network model is trained using the `Nordic Edge AI Lab platform`_.
The whole process how to capture data and train the model is described in the `Nordic Edge AI Lab documentation`_.
You can also see `Gesture recognition use-case demo video`_.

Requirements
************

The application supports the following development kits:

.. table-from-sample-yaml::

Development kits
================

.. tabs::

   .. tab:: Thingy53

      The `Nordic Thingy:53 <Nordic Thingy:53 Multi-protocol IoT prototyping platform_>`_  is an easy-to-use IoT prototyping platform.
      It allows to create prototypes and proofs-of-concept without building custom hardware.
      The Thingy:53 is built around the nRF5340 SoC, a dual-core wireless SoC.
      The processing power and memory size of its dual Arm Cortex-M33 processors enable it to run embedded machine learning (ML) models directly on the device.

      The Thingy:53 also includes multiple integrated sensors, such as environmental sensors, color and light sensors, accelerometers, and a magnetometer.
      It is powered by a rechargeable lithium-polymer (Li-Po) battery that can be charged through USB-C.
      There is also an external 4-pin JST connector compatible with the Stemma, Qwiic, and Grove standards for hardware accessories.

      .. figure:: images/nordic_thingy.jpg
         :alt: Nordic Thingy:53 kit

   .. tab:: nRF54LM20A

      The `nRF54LM20A`_ development kit requires a sensor evaluation board:

      .. list-table::
         :header-rows: 0

         * - .. figure:: images/sensor_EB_top.jpeg
                :alt: Sensor Evaluation Board top view

           - .. figure:: images/sensor_EB_bottom.jpeg
                :alt: Sensor Evaluation Board bottom view

      The Sensor Evaluation Board is connected to the DK through the **EXP** port on the nRF54LM20A board:

      .. figure:: images/nrf54lm20a_EB.jpeg
         :alt: nRF54LM20A sensor EXP port

   .. tab:: nRF54L15 TAG

      The `nRF54L15TAG`_ is a development board for the nRF54L15 SoC.
      It is a small, low-cost development board that is perfect for prototyping and testing.

      .. list-table::
         :header-rows: 0

         * - .. figure:: images/nrf54l15tag_top.jpeg
                :alt: nRF54L15 TAG top view

           - .. figure:: images/nrf54l15tag_bottom.jpeg
                :alt: nRF54L15 TAG bottom view

Sensor BMI270
=============

The `Bosch BMI270`_ is a 3-axis accelerometer and 3-axis gyroscope IMU sensor.
All development kits use the sensor to collect data for gesture recognition.

Setting up software environment
*******************************

For full instructions on preparing your development environment, see :ref:`setting_up_environment`.

User interface
**************

This section describes the user interface available on development kits in this application.

Buttons and LEDs
================

The project has two keyboard control modes: Presentation Control and Music Control.
Depending on the control mode, recognized gestures are mapped to different keyboard keys.
Switch between control modes by pressing the user button.

The following table explains the LED indications for control modes and Bluetooth connection states on each device, and shows which button switches between control modes:

.. list-table:: LED indication in different device states
   :header-rows: 1
   :widths: 8 18 27 27 27

   * - Device
     - Mode switch
     - No Bluetooth connection
     - Presentation Control mode
     - Music Control mode

   * - Thingy:53
     - Press the button on top of the device to switch between Presentation Control and Music Control modes.
     - * LED glows red.

       .. figure:: images/device-led-no-ble-connect_thingy.gif
          :alt: Thingy53 LED red, no BLE connection
     - * LED glows blue.

       .. figure:: images/device-led-ble-connect-presentation-mode_thingy.gif
          :alt: Thingy53 LED blue, presentation mode
     - * LED glows green.

       .. figure:: images/device-led-ble-connect-music-mode_thingy.gif
          :alt: Thingy53 LED green, music mode

   * - nRF54LM20A
     - Press the **BUTTON 0** to switch between Presentation Control and Music Control modes.
     - * **LED0** glows.

       .. figure:: images/device-led-no-ble-connect_nrf54lm20.gif
          :alt: nRF54LM20A LED0, no BLE connection
     - * **LED2** glows.

       .. figure:: images/device-led-ble-connect-presentation-mode_nrf54lm20.gif
          :alt: nRF54LM20A LED2, presentation mode
     - * **LED1** glows.

       .. figure:: images/device-led-ble-connect-music-mode_nrf54lm20.gif
          :alt: nRF54LM20A LED1, music mode

   * - nRF54L15 TAG
     - Press the **BTN1** to switch between Presentation Control and Music Control modes.
     - * LED glows red.

       .. figure:: images/device-led-no-ble-connect_nrf54l15tag.jpeg
          :alt: nRF54L15 TAG LED red, no BLE connection

     - * LED glows blue.

       .. figure:: images/device-led-ble-connect-presentation-mode_nrf54l15tag.jpeg
          :alt: nRF54L15 TAG LED blue, presentation mode

     - * LED glows green.

       .. figure:: images/device-led-ble-connect-music-mode_nrf54l15tag.jpeg
          :alt: nRF54L15 TAG LED green, music mode

Configuration
*************

|config|

.. _app_gesture_recognition_data_collection:

Building firmware for data collection
=====================================

It is possible to create a build that outputs raw data from the accelerometer and gyro sensors on the serial port.
In this mode, you must have an additional development kit running the `Nordic central UART sample`_ to receive the NUS data.
This allows to capture data for training new models and to test and implement new use cases.
The output consists of 16-bit integers separated by a comma, in the following order:

.. code-block::

   <acc_x>,<acc_y>,<acc_z>,<gyro_x>,<gyro_y>,<gyro_z>

Column headers are not included.
The output rate is the configured sampling frequency (default 100 Hz).

1. To build this version, enable the following option in the :file:`prj.conf` file:

   .. code-block::

      CONFIG_DATA_COLLECTION_MODE=y

#. To forward the same data over Bluetooth LE using `Nordic UART Service (NUS)`_, additionally enable:

   .. code-block::

      CONFIG_DATA_COLLECTION_BLE_NUS=y

No inference is performed in this mode.
It is intended to simplify the capture of new datasets.
You can find raw dataset used for model training on the `training dataset`_ page.

Building and running
********************

.. |application path| replace:: :file:`applications/gesture_recognition`

.. include:: /includes/application_build_and_run.txt

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

   Once the device is running, Bluetooth LE advertising starts as a HID device and waits for a connection request from the PC.
   Devices can be connected in the same way as a regular Bluetooth keyboard.

#. Pair the Bluetooth device with your PC.
#. Once connected successfully, the serial port terminal returns the following log messages:

   .. code-block::

      Connected 9C:B6:D0:C0:CE:FC (public)
      Security changed: 9C:B6:D0:C0:CE:FC (public) level 2
      Input CCCD enabled
      Input attribute handle: 0
      Consumer CCCD enabled

   After Bluetooth connection, the device changes LED indication from red to green, or red to blue depending on the keyboard control mode.
   You can now use the device to control media playback or presentation slides by making gestures.

Gestures overview
*****************

This section describes the gesture‑to‑action mapping for Presentation Control and Music Control modes.

.. list-table:: Gestures to keyboard keys mapping
   :header-rows: 1

   * - Gesture
     - Presentation control - blue LED
     - Music control - green LED
   * - Double shake
     - F5
     - Media play/pause
   * - Double tap
     - Escape
     - Media mute
   * - Swipe right
     - Arrow right
     - Media next
   * - Swipe left
     - Arrow left
     - Media previous
   * - Rotation clockwise
     - Not used
     - Media volume up
   * - Rotation counter-clockwise
     - Not used
     - Media volume down

Making gestures
===============

This section shows the correct device orientation and motion for performing supported gestures.
The model was trained with a limited dataset.
For optimal gesture recognition across different users, follow the instructions carefully and train a new model with a larger dataset.

Make sure the default (initial) position of the device matches the following:

.. tabs::

   .. group-tab:: Thingy53

      .. figure:: images/initial_orientation_thingy.gif
         :alt: Initial orientation (Thingy53)

   .. group-tab:: nRF54LM20A

      .. figure:: images/initial_orientation_54lm20.gif
         :alt: Initial orientation (nRF54LM20A)

   .. group-tab:: nRF54L15 TAG

      .. figure:: images/initial_orientation_nrf54l15tag.gif
         :alt: Initial orientation (nRF54L15 TAG)

Follow the images below to make gestures.
For better recognition, use your wrists more than your whole hand.
The gestures are performed with the device in the initial position.
Keep in mind the device orientation, as it is important for the recognition to work correctly.

.. tabs::

   .. group-tab:: Thingy53

      .. list-table:: Swipe right and left
         :header-rows: 1

         * - Swipe right
           - Swipe left
         * - .. figure:: images/swipe_right_thingy.gif
                :alt: Swipe right
           - .. figure:: images/swipe_left_thingy.gif
                :alt: Swipe left

      .. list-table:: Rotation clockwise and counter-clockwise
         :header-rows: 1

         * - Rotation clockwise
           - Rotation counter-clockwise
         * - .. figure:: images/rotation_right_thingy.gif
                :alt: Rotation clockwise
           - .. figure:: images/rotation_left_thingy.gif
                :alt: Rotation counter-clockwise

      .. list-table:: Double shake and double tap
         :header-rows: 1

         * - Double shake
           - Double tap
         * - .. figure:: images/double_shake_thingy.gif
                :alt: Double shake
           - .. figure:: images/double_tap_thingy.gif
                :alt: Double tap

   .. group-tab:: nRF54LM20A

      .. list-table:: Swipe right and left
         :header-rows: 1

         * - Swipe right
           - Swipe left
         * - .. figure:: images/swipe_right_nrf54lm20.gif
                :alt: Swipe right
           - .. figure:: images/swipe_left_nrf54lm20.gif
                :alt: Swipe left

      .. list-table:: Rotation clockwise and counter-clockwise
         :header-rows: 1

         * - Rotation clockwise
           - Rotation counter-clockwise
         * - .. figure:: images/rotation_right_nrf54lm20.gif
                :alt: Rotation clockwise
           - .. figure:: images/rotation_left_nrf54lm20.gif
                :alt: Rotation counter-clockwise

      .. list-table:: Double shake and double tap
         :header-rows: 1

         * - Double shake
           - Double tap
         * - .. figure:: images/double_shake_nrf54lm20.gif
                :alt: Double shake
           - .. figure:: images/double_tap_nrf54lm20.gif
                :alt: Double tap

   .. group-tab:: nRF54L15 TAG

      .. list-table:: Swipe right and left
         :header-rows: 1

         * - Swipe right
           - Swipe left
         * - .. figure:: images/swipe_right_nrf54l15tag.gif
                :alt: Swipe right
           - .. figure:: images/swipe_left_nrf54l15tag.gif
                :alt: Swipe left

      .. list-table:: Rotation clockwise and counter-clockwise
         :header-rows: 1

         * - Rotation clockwise
           - Rotation counter-clockwise
         * - .. figure:: images/rotation_right_nrf54l15tag.gif
                :alt: Rotation clockwise
           - .. figure:: images/rotation_left_nrf54l15tag.gif
                :alt: Rotation counter-clockwise

      .. list-table:: Double shake and double tap
         :header-rows: 1

         * - Double shake
           - Double tap
         * - .. figure:: images/double_shake_nrf54l15tag.gif
                :alt: Double shake
           - .. figure:: images/double_tap_nrf54l15tag.gif
                :alt: Double tap

Dependencies
************

This sample uses the following |NCS| services:

* `Nordic UART Service (NUS)`_

This sample uses the following Zephyr libraries:

* `Logging`_
* `Sensor`_
