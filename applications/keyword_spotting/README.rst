.. _app_keyword_spotting:

Keyword Spotting
################

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

* blinks **LED0** for one second,
* sends a status message on ``UART30``.

.. note::
   The current implementation in this application performs wakeword detection.
   A separate keyword classification stage is not enabled in this revision.

Requirements
************

The application supports the following development kit:

.. table-from-sample-yaml::

The application also requires PDM digital microphone connected to pins specified in :file:`boards/nrf54lm20dk_nrf54lm20a_cpuapp.overlay`.
The application is expecting audio data to be provided on left channel.

The application was tested with an `Adafruit PDM MEMS Microphone`_ module.
It can be powered from ``VDD:IO`` supply.
For this module connect its ``SEL`` pin to ground to select left channel.
Using other microphones may require adapting configuration parameters of PDM in :file:`src/dmic.c`.

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

.. |application path| replace:: :file:`applications/keyword_spotting`

.. include:: /includes/application_build_and_run.txt

Testing
=======

|test_application|

#. |connect_kit|
#. |connect_terminal_kit| The application is using both serial ports.
#. Open one terminal for Zephyr logs and one for control output from ``UART30``.
#. Say the wakeword phrase "Okay Nordic".
#. Observe a one-second blink on **LED0** and messages similar to the following:

.. code-block:: console

   Waiting for wakeword
   Wakeword detected

Dependencies
************

This application uses the following |EAI| libraries:

* :ref:`nrf_edgeai_lib`

It uses the following Zephyr libraries:

* `Logging`_
* Audio (DMIC)
* UART Driver

API documentation
*****************

.. doxygengroup:: keyword_spotting
