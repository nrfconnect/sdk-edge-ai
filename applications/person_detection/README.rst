.. _app_person_detection:

Person detection
################

.. contents::
   :local:
   :depth: 2

This application demonstrates real-time person detection from a camera stream using model inference on the Axon NPU.

Application overview
********************

The Person Detection application captures images from an `Arducam Mega camera <Arducam Mega 5MP_>`_ module and processes them through a neural network model running on the Axon NPU.
The camera captures an image with a resolution of 128×128 pixels.
This image is later padded with neutral gray color to match the model input size of 128×160 pixels.
The application converts the camera input to the model format, runs inference, and post-processes the output to extract bounding boxes with confidence scores.

The application uses the ``person_det`` model from `MCUnet`_.
It uses three detection heads with different spatial scales to identify objects at various sizes.
Detected bounding boxes are post-processed using Non-Maximum Suppression (NMS) to remove overlapping detections and filter low-confidence predictions.

The application provides visual feedback through :ref:`LEDs <app_person_detection_ui>` for image capture and current detection status.

The application continuously captures frames at regular intervals, processes them through the model, and logs any detected objects along with their bounding box coordinates and confidence scores.

Requirements
************

The application supports the following development kit:

.. table-from-sample-yaml::

The application also requires an SPI camera and was tested with `Arducam Mega 5MP (B401) camera <Arducam Mega 5MP_>`_.
Configure the development kit using `Board Configurator`_ to provide 3.3V to power the camera.

Pin mapping
===========

See the following table for the camera-to-DK pin mapping:

.. list-table::
   :header-rows: 1

   * - Description
     - Arducam Mega Pin
     - nRF54LM20 DK Pin
   * - Power supply (3.3V)
     - ``VCC``
     - ``VDD:IO``
   * - Ground
     - ``GND``
     - ``GND``
   * - Chip select
     - ``CS``
     - ``P1.7``
   * - SPI MOSI
     - ``MOSI``
     - ``P1.6``
   * - SPI MISO
     - ``MISO``
     - ``P1.5``
   * - SPI Clock
     - ``SCK``
     - ``P1.4``

For detailed pin configuration, refer to the device tree overlay :file:`boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay` file.

.. _app_person_detection_ui:

User interface
**************

This section describes the user interface of the application.

LEDs
====

LED0 (capture LED):
   Toggles (changes state between on and off) each time a frame is captured from the camera and processed.

LED1 (detection LED):
   Turns on when persons are detected in the frame.
   Turns off when no detections are found.

Configuration
*************

|config|

Configuration options
=====================

|application_kconfig|

.. options-from-kconfig::
   :show-type:

Building and running
********************

.. |application path| replace:: :file:`applications/person_detection`

.. include:: /includes/application_build_and_run.txt

Testing
=======

|test_application|

#. |connect_kit|
#. |connect_terminal_kit|
#. Observe **LED0** toggling on each frame capture.
#. Place a person in front of the camera module.
#. Observe **LED1** changing state when detections occur.
#. Check the serial output for bounding box coordinates and confidence scores.

Application output
==================

The application shows the following output:

.. code-block:: console

   [00:00:01.070,782] <inf> main: Person detection start
   [00:00:01.211,020] <inf> main: No detections
   [00:00:01.721,870] <inf> main: No detections
   [00:00:02.200,188] <inf> main: Bounding box 0: head s16, [59.1, 58.9, 121.1, 125.6] score 0.403
   [00:00:02.713,248] <inf> main: Bounding box 0: head s32, [37.5, 43.8, 131.0, 124.6] score 0.554

Dependencies
************

This application uses the following |EAI| library:

* :ref:`lib_axon`

This application uses the following Zephyr libraries:

* `Logging`_
* `Video`_
