.. _app_ww_kws:

Wakeword and Keyword Spotting
#############################

.. contents::
   :local:
   :depth: 2

This application demonstrates wakeword detection from a digital microphone stream using |EAILib| and Axon-based inference.

Application overview
********************

The application samples single-channel 16 kHz audio from a PDM microphone and feeds it to an nRF Edge AI wakeword model.
The wakeword phrase used by the bundled model is "Okay Nordic".
To reduce false detections, model output is postprocessed with a predictions history window.
Parameters used for postprocessing are prediction probability threshold, predictions history length and number of predictions above threshold in predictions history.
When a wakeword is detected, the application:

* Blinks **LED0** for one second.
* Sends a status message on UART30.

You can easily replace the bundled wakeword with a custom one using the Text to Wake Word feature of the `Nordic Edge AI Lab`_.

.. note::
   The current implementation in this application performs wakeword detection.
   A separate keyword classification stage will be added in future releases.

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
   Blinks for one second when the wakeword is detected.

UART30:
   Prints wakeword state messages:

   * ``Waiting for wakeword``
   * ``Wakeword detected``

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

.. |application path| replace:: :file:`applications/ww_kws`

.. include:: /includes/application_build_and_run.txt

Testing
=======

|test_application|

#. |connect_kit|
#. |connect_terminal_kit| The application is using both serial ports.
#. Open one terminal for Zephyr logs and one for control output from UART30.
#. Say the wakeword phrase "Okay Nordic".
#. Observe a one-second blink on **LED0** and the following messages:

.. code-block:: console

   Waiting for wakeword
   Wakeword detected

Dependencies
************

This application uses the following |EAI| library:

* :ref:`nrf_edgeai_lib`

This application uses the following Zephyr libraries:

* `Logging`_
* Audio (DMIC)
* UART Driver

API documentation
*****************

.. doxygengroup:: ww_kws
