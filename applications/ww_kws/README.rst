.. _app_ww_kws:

Wakeword and Keyword Spotting
#############################

.. contents::
   :local:
   :depth: 2

This application demonstrates wakeword detection and keyword spotting from a digital microphone stream using |EAILib| and Axon-based inference.
It also features integrated model observability, allowing you to monitor and analyze inference metrics during operation.

Application overview
********************

The application samples single-channel 16 kHz audio from a PDM microphone and feeds it to nRF Edge AI models.
The wakeword phrase used by the bundled model is "Okay Nordic".
In wakeword detection stage, model output is postprocessed with a predictions history window.
Parameters used for postprocessing are prediction probability threshold, predictions history length and number of predictions above threshold in predictions history.
When a wakeword is detected, the application switches to keyword detection stage.

The bundled keyword spotting model supports the following keywords:

* Go
* Stop
* Up
* Down
* Yes
* No
* On
* Off
* Right
* Left

In the keyword spotting stage, the application smooths class probabilities using an exponential moving average and reports a detection when class-specific criteria are met.
After period without any keywords spotted the application switches back to wakeword stage.

You can also configure the application to stay in one of these stages using the application-specific Kconfig options.

Observability
=============

The application collects runtime observability metrics using the :ref:`nrf_edgeai_obsv_lib`.
Enable the ``CONFIG_MODELS_OBSERVABILITY`` Kconfig option to wire metrics into the bundled models.
Each model stage owns its observability context and Memfault transport binding in the :file:`src/ww/wakeword.c` and :file:`src/kws/kws.c` files.

The wakeword stage only registers the :ref:`nrf_edgeai_obsv_metrics_built_in_probability` metric.
The keyword spotting stage registers the :ref:`nrf_edgeai_obsv_metrics_built_in_probability` and :ref:`nrf_edgeai_obsv_metrics_built_in_transition` metrics.

Metrics are collected every 24 hours (see the ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT`` Kconfig option) and sent to the `Memfault`_ using `Custom Data Recording <Memfault Custom Data Recording_>`_ registered by the :ref:`nrf_edgeai_obsv_lib`.

Replacing models
================

You can replace the bundled models using the `Text to Wake Word Detection <Nordic Edge AI Lab Wake Word Detection_>`_ feature of the `Nordic Edge AI Lab`_ or one of `ready-to-use models <Nordic Edge AI Lab ready-to-use models_>`_.

With :kconfig:option:`CONFIG_APP_MODEL_OTA` enabled (the default - see "Model-only OTA update" below), replacing a model at runtime is just a matter of flashing a new model package to the relevant partition; the steps below (replacing the generated header and rebuilding) are only needed to change which model the application *builds packages from* in the first place, or to restore this application's original pre-model-OTA behavior (:kconfig:option:`CONFIG_APP_MODEL_OTA` disabled).

.. tabs::

   .. group-tab:: Wakeword detection model

      To replace the model used in wakeword detection stage complete the following steps:

      1. Replace the files inside :file:`src/ww/nrf_edgeai_generated` directory with files downloaded from `Nordic Edge AI Lab`_.
      #. If the new model's ``persistent_vars`` array size differs from the previous one, update :kconfig:option:`CONFIG_APP_WW_PERSISTENT_VARS_SIZE` to match (see :file:`src/ww/nrf_edgeai_generated/nrf_edgeai_user_model_axon.h`'s own ``axon_model_<name>_persistent_vars[]`` array size).
      #. Adjust the Kconfig options to tune wakeword detection postprocessing to your model.


   .. group-tab:: Keyword spotting model

      To replace the model used in keyword spotting stage complete the following steps:

      1. Replace the files inside :file:`src/kws/nrf_edgeai_generated` directory with files downloaded from `Nordic Edge AI Lab`_.
      #. If the new model's ``persistent_vars`` array size differs from the previous one, update :kconfig:option:`CONFIG_APP_KWS_PERSISTENT_VARS_SIZE` to match (see :file:`src/kws/nrf_edgeai_generated/nrf_edgeai_user_model_axon.h`'s own ``axon_model_<name>_persistent_vars[]`` array size).
      #. Update the ``keyword_detection_ctxs`` array in the :file:`src/kws/kws.c` file with keyword labels from :file:`src/kws/nrf_edgeai_generated/nrf_edgeai_user_model_labels.h` file and thresholds for keyword spotting.
      #. When using the observability feature, set the ``CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES`` Kconfig option to number of keywords spotted plus 2 for auxiliary classes.
      #. Adjust the Kconfig options to tune keyword spotting postprocessing to selected model.

.. _app_ww_kws_model_ota:

Model-only OTA update
======================

This application does not use mcuboot, so its second application slot (``slot1_partition``) is unused on the boards it supports.
The board overlay in :file:`applications/ww_kws/boards/` repurposes that space as two dedicated partitions instead - ``model_storage_ww`` and ``model_storage_kws``, one per model - sized to comfortably fit larger models too.
At boot, the application loads and validates each model's own package from its partition and wires it up for inference - see :ref:`lib_model_ota` for how the package format, host-side packaging tools, and on-device loading work, including how CPU op extensions and ``persistent_vars`` (both used by the bundled models) are handled.
Flashing a new package to either partition is enough to change what that model predicts, without rebuilding or reflashing the application.

Making model OTA optional
--------------------------

Model-only OTA is enabled by default (:kconfig:option:`CONFIG_APP_MODEL_OTA` defaults to ``y``).
Build with it disabled to restore this application's original, pre-model-OTA behavior instead: both models are compiled directly into the application image, and no ``model_storage_ww``/``model_storage_kws`` partitions or flash packages are involved.

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20b/cpuapp applications/ww_kws -- -DCONFIG_APP_MODEL_OTA=n

Packaging and flashing a model
--------------------------------

Both models' packages are built automatically as part of a normal application build - no separate build or manual packaging step is needed:

.. code-block:: console

   west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build applications/ww_kws

This produces ``build/ww_kws/ww_model_pkg.bin``/``.hex`` and ``build/ww_kws/kws_model_pkg.bin``/``.hex``.
Flash the application as usual, then flash either (or both) model packages to their partitions:

.. code-block:: console

   nrfutil device program --firmware build/ww_kws/ww_model_pkg.hex --core Application \
     --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_NONE
   nrfutil device program --firmware build/ww_kws/kws_model_pkg.hex --core Application \
     --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

``reset=RESET_SYSTEM`` on the last command ensures the board resumes execution automatically; without it, ``nrfutil`` leaves the CPU halted after flashing.
Repeat after replacing a model (see "Replacing models" above) and rebuilding, to observe the new model's behavior - no need to reflash anything but that one model's package.

Requirements
************

The application supports the following development kit:

.. table-from-sample-yaml::

The application also requires PDM digital microphone connected to the pins specified in the :file:`boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay` file.
The application is expecting audio data to be provided on left channel.

Pin mapping
===========

The application was tested with an `Adafruit PDM MEMS Microphone`_ module.
It can be powered from 1.8V ``VDD:IO`` supply.
The following table show how to connect this module to the DK:

.. list-table::
   :header-rows: 1

   * - Adafurit DMIC
     - nRF54LM20 DK
   * - ``3V``
     - ``VDD:IO``
   * - ``GND``
     - ``GND``
   * - ``SEL``
     - ``GND``
   * - ``CLK``
     - ``P1.4``
   * - ``DAT``
     - ``P1.5``

The ``SEL`` pin is responsible for selecting audio channel and connecting it to ground selects left channel.

To use other microphones, adapt configuration parameters of PDM in the :file:`src/dmic.c` file.

User interface
***************

LED0:
   Behavior depends on the selected application mode:

   * Lit when keyword spotting stage is active in wakeword-gated keyword spotting mode.
   * Blinks for one second when wakeword is detected in wakeword only mode.
   * Stays off in keyword spotting only mode.

LED1:
   Blinks for one second on each keyword spotted.

UART30 (VCOM0):
   Prints runtime state messages:

   * ``Waiting for wakeword``
   * ``Wakeword detected``
   * ``Waiting for keywords``
   * ``Keyword spotted: <name>``
   * ``Keyword spotting window timeout``

Configuration
*************

|config|

Configuration options
=====================

|application_kconfig|

.. options-from-kconfig::
   :show-type:

Additional configuration
========================

Check and configure the following library options that are used by the application:

* ``CONFIG_MEMFAULT_NCS_PROJECT_KEY`` - A key to your `Memfault`_ project
* ``CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES`` - Number of classes the observed model predicts
* ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC`` - Interval of automatic metrics collection.
  For testing, use a shorter interval and enable server debug mode for the device in the  `Memfault`_ dashboard.

Build types
===========

For build type overview and guidance on how to run them, see `Custom build types`_ and `Providing CMake options`_.
The application supports the following build types:

.. list-table:: Wakeword and Keyword Spotting build types
   :widths: auto
   :header-rows: 1

   * - Build type
     - File name
     - Description
   * - Debug (default)
     - :file:`prj.conf`
     - Debug version of the application with logging and assertions enabled.
   * - Release
     - :file:`prj_release.conf`
     - Release version of the application with logging disabled and compiler optimizations.

Configuration files
===================

The application provides predefined :file:`observability.conf` configuration file for enabling observability.
This file enables the ``CONFIG_MODELS_OBSERVABILITY`` Kconfig option, Bluetooth LE, Memfault Diagnostic Service and required dependencies to provide observability for bundled models.

Check `Providing CMake options`_ and use :makevar:`EXTRA_CONF_FILE` variable to include this configuration file.

.. note::
   Set the ``CONFIG_MEMFAULT_NCS_PROJECT_KEY`` Kconfig option to your `Memfault`_ project key to successfully send metrics to Memfault.

Building and running
********************

.. |application path| replace:: :file:`applications/ww_kws`

.. include:: /includes/application_build_and_run.txt

Testing
=======

|test_application|

#. |connect_kit|
#. |connect_terminal_kit| The application is using both serial ports.
#. Open one terminal for Zephyr logs and one for control output from UART30.
#. Say the wakeword phrase "Okay Nordic".
#. Observe **LED0** light up.
#. Say one of the bundled model keywords (for example, "Yes" or "No").
#. Observe that **LED1**  blinks for one second.
#. Stop speaking and wait for timeout.

If you have enabled the model observability, also complete the following steps:

#. Connect gateway to your device.
   Ready-to-use gateways are `nRF Connect Device Manager`_ or `Memfault WebBluetooth Client`_.
   Check `Peripheral MDS sample <Peripheral MDS sample: Testing_>`_ for details.
#. Wait for a duration set by the ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT_INTERVAL_SEC`` Kconfig option.
#. In a web browser, navigate to `Memfault`_.
#. In the left-hand menu, go to :guilabel:`CDR Payloads`.
#. Select a CDR payload from your test device with "edgeai_observability" as reason.
   You can filter the list of CDR payloads by device name and time.
#. Download the payload for metrics analysis.
#. Decode it on a host PC using :file:`scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py` script in binary mode:

   .. code-block:: shell

      ./scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py --binary --file <downloaded>.bin

   See :ref:`nrf_edgeai_obsv_script` for installation and other input modes.
   Payload format is defined in the :file:`lib/nrf_edgeai_obsv/obsv.cddl` file.

Application output
==================

The application shows the following output from UART30 (VCOM0):

.. code-block:: console

   Waiting for wakeword
   Wakeword detected
   Waiting for keywords
   Keyword spotted: Yes
   Keyword spotting window timeout

Dependencies
************

This application uses the following |EAI| library:

* :ref:`nrf_edgeai_lib`
* :ref:`nrf_edgeai_obsv_lib`

This application uses the following Zephyr libraries:

* `Logging`_
* Audio (DMIC)
* UART Driver

API documentation
*****************

.. doxygengroup:: ww_kws
