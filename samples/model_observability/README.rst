.. _model_observability_sample:

Model observability sample
##########################

.. contents::
   :local:
   :depth: 2

This sample demonstrates how to send runtime statistics of a machine learning model to the cloud using Memfault.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

The sample uses only the blue channel of the on-board RGB LED.

Overview
********

The sample does not run an activity recognition model; it feeds predefined class-probability vectors to the :ref:`nrf_edgeai_obsv_lib` in the same format as model inference output.
The sample validates the end-to-end observability path from the device through `Memfault`_ to the decoding script, using the predefined probability vectors as input.
For a conceptual introduction to model observability, see :ref:`model_observability`.
For integration details, refer to the :ref:`nrf_edgeai_obsv_lib` documentation.

The probability vectors cover the following activity classes, in the following order:

1. Walking
2. Sitting
3. Running
4. Laying

The sample feeds these vectors as three fixed sequences.
In the source code and in this documentation page, each sequence is called a scenario.

Predefined class-probability sequences
**************************************

The sequences are defined in :file:`src/scenarios.c`.
For each vector, the activity with the highest probability is the predicted class.
The following table summarizes predicted activities for each scenario:

.. list-table:: Predicted activities per scenario
   :header-rows: 1
   :widths: 7 7 10 10 10 10 46

   * - Scenario
     - Vectors
     - Walking
     - Sitting
     - Running
     - Laying
     - Order of the activities
   * - 1
     - 20
     - 7 (35%)
     - 3 (15%)
     - 1 (5%)
     - 9 (45%)
     - Walking → running → walking → sitting → laying
   * - 2
     - 20
     - 3 (15%)
     - 3 (15%)
     - 1 (5%)
     - 13 (65%)
     - Laying → sitting → laying → sitting → walking → running
   * - 3
     - 20
     - 5 (25%)
     - 7 (35%)
     - 3 (15%)
     - 5 (25%)
     - Sitting → walking → running → walking → sitting → laying

The probabilities of the predicted class vary between the vectors, and the vectors at the activity changes have lower probabilities, so the certainty metrics are not constant.

The sample feeds one vector every 500 milliseconds, set by the ``CONFIG_APP_FEED_INTERVAL_MS`` Kconfig option.
After feeding a scenario, the sample waits until the automatic collection encodes it into a recording, then starts the next scenario.
After the third scenario, the sample starts again from the first one.

Metrics collection and Memfault upload
======================================

The sample collects the following output metrics from each probability vector:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Metric
     - Description
   * - :ref:`Class predictions distribution <nrf_edgeai_obsv_metrics_built_in_class_pred>`
     - How often each class is predicted and with what probability, including how long the same class is predicted in a row.
   * - :ref:`Transition matrix <nrf_edgeai_obsv_metrics_built_in_transition>`
     - How often the predicted class changes from one class to another.
   * - :ref:`Model certainty descriptor <nrf_edgeai_obsv_metrics_built_in_certainty>`
     - How confident the model is in its predictions.

The observability code is located in the :file:`src/model_obsv.c` file.
It creates the observability context, registers the metrics and binds the Memfault transport.
The :file:`src/main.c` file defines the model metadata, with a fixed model ID of ``777``, and passes it to the observability code.

The metrics are collected automatically at an interval set by the ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC`` Kconfig option.
After each collection, the metrics are encoded and staged as a `Memfault Custom Data Recording <Memfault Custom Data Recording_>`_ (CDR).
A gateway, for example a phone, reads the staged recording over the Memfault Diagnostic Service (MDS) and uploads it to `Memfault`_.
The metrics are reset after each collection, so every recording describes one collection period.

Each recording contains exactly one scenario:

* The sample starts the next scenario only after the previous one is staged as a recording.
* A new collection is skipped while the previous recording is still waiting to be read by the gateway.
  The next scenario then stays in the metrics until the gateway reads the previous recording.
* A whole scenario is fed within one collection interval, which is checked at build time.

User interface
**************

This section describes the user interface available on development kits in this sample.

LEDs
====

The on-board **RGB LED** (status LED) blinks blue and shows the Bluetooth® Low Energy connection state.
The firmware drives the blue channel through **LED0**.

* Fast blinking (3 Hz) - The device is advertising and waits for a Bluetooth LE connection.
* Slow blinking (1 Hz) - A peer is connected.

Configuration
*************

|config|

Configuration options
=====================

|sample_kconfig|

.. options-from-kconfig::
   :show-type:

Additional configuration
========================

For the observability library, Memfault transport, built-in metrics, and buffer sizing, see :ref:`nrf_edgeai_obsv_lib`.
For the Bluetooth® LE transport (MDS) options, see :ref:`nrf:peripheral_mds`.

The :file:`prj.conf` file of the sample enables the library, the Memfault transport with automatic collection, and MDS.
The following settings are specific to the sample:

* ``CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES=4`` - The number of the activity classes.
  The sample returns an error at start if the model has more classes.
* ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC=120`` - A short collection interval, so that the three scenarios are sent within a few minutes.
  The sample checks at build time that a whole scenario is fed within this interval.
  Memfault accepts a limited number of recordings per device per day, so enable server debug mode for the device in the `Memfault`_ dashboard.
* ``CONFIG_MEMFAULT_PROJECT_KEY`` - Your `Memfault`_ project key.
  The key in the :file:`prj.conf` file is a placeholder, and your device data is not available in your own Memfault project until you replace it.
* ``CONFIG_MEMFAULT_NCS_DEVICE_ID`` - A fixed device ID, so that you can find the device in the `Memfault`_ dashboard.

Building and running
********************

.. |sample path| replace:: :file:`samples/model_observability`

.. include:: /includes/build_and_run.txt

Testing
=======

|test_sample|

1. |connect_kit|
#. |connect_terminal_kit|
   Connect to the serial device printing console output.
#. Observe that the RGB LED blinks blue at 3 Hz and that the terminal displays messages similar to the following:

   .. code-block:: console

      Observability initialized: model id 777, 4 classes
      Scenario 1: feeding 20 probability vectors
      Scenario 1 fed: walking 7, sitting 3, running 1, laying 9
      Scenario 1: waiting for auto collect

#. Connect a gateway to your device.
   Ready-to-use gateways are `nRF Connect Device Manager`_ or `Memfault WebBluetooth Client`_.
   Check the Testing section of :ref:`nrf:peripheral_mds` for details.
   After the connection, the RGB LED blinks at 1 Hz.
#. Wait until the terminal shows that the three scenarios are staged:

   .. code-block:: console

      Scenario 1 staged as CDR
      Scenario 2: feeding 20 probability vectors
      Scenario 2 fed: walking 3, sitting 3, running 1, laying 13
      Scenario 2: waiting for auto collect
      Scenario 2 staged as CDR
      Scenario 3: feeding 20 probability vectors
      Scenario 3 fed: walking 5, sitting 7, running 3, laying 5
      Scenario 3: waiting for auto collect
      Scenario 3 staged as CDR

#. In a web browser, open `Memfault`_.
#. In the left-hand menu, go to :guilabel:`CDR Payloads`.
#. Select a CDR payload from your test device with "edgeai_observability" as reason.
   Filter the list of CDR payloads by device name and time.
#. Download the payload for metrics analysis.
#. Decode it on a host PC using the :file:`scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py` script in binary mode:

   .. code-block:: shell

      ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --binary --file <downloaded>.bin

   See :ref:`nrf_edgeai_obsv_script` for installation and other input modes.
   The payload format is defined in the :file:`lib/nrf_edgeai_obsv/obsv.cddl` file.

#. Compare the decoded metrics with the scenario.
   For example, ``num_inferences`` is 20, the class predictions distribution matches the predicted activities in the scenarios table above, and the transition matrix shows the order of the activities.

Dependencies
************

This sample uses the following |EAI| library:

* :ref:`nrf_edgeai_obsv_lib`

This sample uses the following |NCS| service:

* :ref:`nrf:peripheral_mds`

This sample uses the following Zephyr library:

* :ref:`zephyr:logging_api`