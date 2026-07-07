.. _data_forwarder_sample:

Data forwarder
##############

.. contents::
   :local:
   :depth: 2

This sample streams real sensor measurements from a development kit to a host computer.
It uses a CBOR-based framing protocol for :ref:`data_forwarder_host_tool`, with an optional ASCII mode for `Edge Impulse's data forwarder`_ CLI compatibility.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

When using Bluetooth LE transport, your host must be able to connect to the peripheral directly.
If it cannot connect, you need a second development kit running the `Nordic central UART sample`_ as a USB serial bridge.

The default configuration on the nRF54L15 TAG reads a 6-axis BMI270 IMU and appends environmental data from an on-board BME688 sensor.

Overview
********

The sample periodically reads data from an on-board sensor and forwards it to a host over `Nordic UART Service (NUS)`_ (default) or UART.
By default, the sample uses CBOR messages wrapped in COBS framing, as defined in :file:`cddl/data_forwarder.cddl`.
Session metadata is sent periodically so a host can join an active stream.

The sample performs the following operations:

* Initializes the selected sensor driver and starts sampling at a fixed rate.
* Encodes each sample and sends it through the configured transport.

Transport
=========

The sample can send protocol frames to the host over Bluetooth LE or UART.
Select the transport using the ``DATA_FWD_TRANSPORT`` Kconfig option.

Bluetooth LE NUS
   Sample uses `Nordic UART Service (NUS)`_ for sending the protocol frames.

UART transport
   Sample uses UART for sending the protocol frames.

   .. note::
      UART transport is not available on the nRF54L15 TAG device without any external UART-to-USB converter.

Increasing throughput
=====================

If you need higher streaming bandwidth, apply the following adjustments:

Bluetooth LE with direct host connection
   When using :ref:`data_forwarder_host_tool` over Bluetooth LE on the host:

   #. Disconnect or move other Bluetooth devices off the same adapter.
      A single controller shares radio airtime across all active links, which reduces stream bandwidth.
   #. Follow the Bluetooth LE throughput tuning procedure in :file:`tools/data_forwarder_host/docs/ble_throughput_tuning.md`.

Bluetooth LE with Central UART on a second DK
   On the development kit running the `Nordic central UART sample`_ sample, set the following options in :file:`prj.conf` to increase the NUS notification payload size:

   .. code-block:: ini

      CONFIG_BT_L2CAP_TX_MTU=247
      CONFIG_BT_BUF_ACL_RX_SIZE=251

   The Data forwarder sample sets matching MTU values on the peripheral side.
   You may also increase the Central UART baud rate in its board overlay (for example, ``current-speed = <921600>;`` on the relay UART node).

UART
   Increase the baud rate of the data UART by setting the ``current-speed`` property in the board overlay.
   For example, to use 921600 baud:

   .. code-block:: devicetree

      &uart20 {
         current-speed = <921600>;
      };

Protocol
********

The sample encodes sensor readings into protocol frames before sending them to the host.
Zephyr sensor drivers return values stored in ``struct sensor_value``.
These are typically physical values converted by driver from raw sensor readings.

Data forwarder protocol supports float and integer sample formats.
Sensor wrappers provided by the sample return physical value as a float, or as an integer in micro units, depending on the ``CONFIG_DATA_FWD_PROTO_INT32_VALUES`` Kconfig option.

Configuration
*************

|config|

|sample_kconfig|

.. options-from-kconfig::
   :show-type:

Adapting the sample for other boards
====================================

To run the sample on a board that is not listed in :file:`sample.yaml`, add a devicetree overlay for that board target.
See the following guides for the general workflow:

* `Configuring devicetree`_
* `Devicetree overlays`_

For this sample, you typically need to:

#. Add a board overlay under :file:`boards/<board>_<soc>_<variant>.overlay`.
#. Configure or enable the required sensor device tree nodes (``bmi270``, ``adxl367``, ``bme688``, or your own sensor).
#. When using UART transport, set the ``ncs,data-forwarder-uart`` chosen node to the UART used for data forwarding and assign a separate console output (for example, RTT or another UART).
#. Update :file:`sample.yaml` with the new board target when adding official support.

Adapting the sample for other sensors
=====================================

To forward data from a sensor that is not supported out of the box, implement the sensor wrapper API:

.. doxygengroup:: sample_data_fwd_sensor

Use :file:`src/sensor/bmi270.c`, :file:`src/sensor/adxl367.c`, or :file:`src/sensor/bme688.c` as references.
Then:

#. Add a new ``config DATA_FWD_SENSOR_*`` entry to the ``DATA_FWD_SENSOR`` option in :file:`Kconfig`.
#. Add the source file to :file:`CMakeLists.txt` under the corresponding ``CONFIG_DATA_FWD_SENSOR_*`` block.
#. Set a ``CONFIG_DATA_FWD_PROTO_MAX_CHANNELS`` default for the new channel count in :file:`Kconfig`.

Host-side setup
===============

Complete the host-side setup for your chosen data collection workflow before building and running the sample.

.. tabs::

   .. group-tab:: Data forwarder host tool

      The :ref:`data_forwarder_host_tool` decodes CBOR/COBS frames from the default sample build and is intended for use with `Nordic Edge AI Lab`_ data collection workflows.

      Before running the sample, get the :ref:`data_forwarder_host_tool`.

   .. group-tab:: Edge Impulse CLI

      The sample outputs comma-separated sensor values compatible with the ``edge-impulse-data-forwarder`` CLI tool when the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option is enabled.
      This replaces CBOR binary framing with CSV lines (one sample per line, values separated by commas, terminated with a CRLF line ending) compatible with the protocol specified by `Edge Impulse's data forwarder`_.

      Before running the sample, complete the following setup:

      #. :ref:`Set up Edge Impulse <setup_edge_impulse>`.
      #. Prepare your own project using `Edge Impulse studio`_ external web tool.
         See :ref:`edge_impulse_integration` for more information on using the tool.
      #. Add the following options to sample configuration:

         .. code-block:: ini

            CONFIG_DATA_FWD_PROTO_ASCII_MODE=y
            CONFIG_DATA_FWD_PROTO_CRC=n
            CONFIG_DATA_FWD_PROTO_INT32_VALUES=n

      #. Use a second development kit running the `Nordic central UART sample`_ sample as a serial bridge.

Building and running
********************

.. |sample path| replace:: :file:`samples/data_forwarder`

.. include:: /includes/build_and_run.txt

Testing
=======

|test_sample|

#. |connect_kit|
#. Open an RTT viewer (for example, in |nRFVSC|) to view console output.
   After programming, you should see output similar to the following:

   .. parsed-literal::
      :class: highlight

      transport: BLE NUS transport ready
      data_forwarder: Data forwarder started (sid 8779774)
      transport: BLE connected

Collecting data on the host
===========================

Use one of the following workflows to capture sensor data on your host computer:

.. tabs::

   .. group-tab:: Data forwarder host tool

      The :ref:`data_forwarder_host_tool` decodes CBOR/COBS frames from the default sample build and is intended for use with `Nordic Edge AI Lab`_ data collection workflows.

      To collect data with the default configuration:

      #. Run :ref:`data_forwarder_host_tool` on the host computer and connect to the device over Bluetooth LE.

         Alternatively, if the host has no suitable Bluetooth LE adapter, use a second development kit running the `Nordic central UART sample`_ as a serial bridge.
         Flash Central UART on the second kit, wait for it to connect to the peripheral, then connect the host tool to the Central UART USB serial port.

      #. Enter a label and start a sampling session.
         For details refer to the :ref:`data_forwarder_host_tool` documentation.

   .. group-tab:: Edge Impulse CLI

      To collect data:

      #. Identify data interface.
         With the default Bluetooth LE setup, this is the Central UART bridge USB serial port.
         If you enabled UART transport, use the dedicated data UART instead.
         Do not open that port in a terminal when ``edge-impulse-data-forwarder`` is running.
      #. Run the ``edge-impulse-data-forwarder`` command line tool.
         The tool connects the device to your Edge Impulse project.
         When prompted for a UART port, provide the port carrying sensor data, not the console.
         See the `Edge Impulse's data forwarder`_ documentation for guidance.
      #. Trigger sampling data from the device using `Edge Impulse studio`_:

         a. Go to the :guilabel:`Data acquisition` tab.
         #. In the :guilabel:`Collect data` panel, set the desired values and click :guilabel:`Start sampling`.

            .. figure:: ../edge_impulse/data_forwarder/images/ei_data_acquisition.png
               :scale: 80 %
               :alt: Sampling under Data acquisition in Edge Impulse studio

               Sampling under Data acquisition in Edge Impulse studio

         #. Observe the received sample data on the raw data graph under the panel.

            .. figure:: ../edge_impulse/data_forwarder/images/ei_start_sampling.png
               :scale: 80 %
               :alt: Sampling example

               Sampling example

Dependencies
************

This sample uses the following |NCS| libraries:

* `Nordic UART Service (NUS)`_

This sample uses the following Zephyr libraries:

* `Logging`_
* `Sensor`_
* `UART`_

The data forwarder protocol uses the following Zephyr libraries:

* `CRC`_
* `ZCBOR`_
* `Ring Buffers`_
* :file:`include/zephyr/data/cobs.h`
