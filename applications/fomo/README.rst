.. _axon_fomo_demo:

Axon FOMO demo
##############

.. contents::
   :local:
   :depth: 2

The Axon FOMO demo demonstrates use of the Axon to accelerate inference of the Axon-compatible neural model.

Requirements
************

The sample supports the following development kit:

.. table-from-rows:: /includes/sample_board_rows.txt
   :header: heading
   :rows: nrf54lm20dk_nrf54lm20b

The sample also requires an Arducam MEGA SPI camera and configuring DK VDD to 3.3V with `Board Configurator app`_.

Overview
********

The sample can serve as showcase for the Axon accelerator.

The sample runs FOMO object detection model on the Axon accelerator.
It uses images from the connected camera to detect beer bottles and soda cans.

As the Arducam Mega SPI Camera is not yet supported in Zephyr, the sample contains also minimal driver for it.

The Axon-compatible model is present in the :file:`model` directory.
It was then compiled with Axon Compiler and stored in :file:`model/outputs` subdirectory.

Objects tracking
================

The sample implements simple objects tracking.
It requires at least 3 detections of the same type of object in similar area.
The object is dropped from tracking when it was missing for 3 consecutive predictions.

Wiring
******

Connect camera to DK as in following table

.. list-table:: SPI Camera connections.
   :header-rows: 1

   * - SPI Camera
     - nRF54LM20 DK
   * - VCC
     - VDDIO
   * - GND
     - GND
   * - SCK
     - P1.4
   * - MISO
     - P1.5
   * - MOSI
     - P1.6
   * - CS
     - P1.7

Building and running
********************

.. |sample path| replace:: :file:`applications/fomo`

.. include:: /includes/build_and_run.txt

Testing
=======

|test_sample|

#. |connect_kit|
#. |connect_terminal|
#. Point the camera at bottle or can.
#. Observe the terminal output for the model predictions.

Sample output
==============

Prediction results are logged on terminal:

.. code-block:: console

   [00:00:37.115,423] <inf> fomo: New prediction results:
   [00:00:37.120,965] <inf> fomo: can (0.85) [x: 8, y: 6, width: 3, height: 1]
   [00:00:37.128,314] <inf> fomo: Tracked results:
   [00:00:37.133,235] <inf> fomo: can (0.85) [x: 8, y: 6, width: 3, height: 1]
   [00:00:37.140,583] <inf> fomo: beer (0.24) [x: 9, y: 6, width: 1, height: 1]

Troubleshooting
================

If the model does not detect any object try moving the object in camera's filed of view.
If the camera driver is reporting ``Capture timeout!`` the camera is probably broken.
