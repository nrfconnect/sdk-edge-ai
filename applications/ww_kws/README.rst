.. _app_ww_kws:

Wakeword and Keyword Spotting
#############################

.. contents::
   :local:
   :depth: 2

This application demonstrates wakeword detection and keyword spotting from a digital microphone stream using |EAILib| and Axon-based inference.

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

Replacing models
================

You can replace the bundled models using the `Text to Wake Word Detection <Nordic Edge AI Lab Wake Word Detection_>`_ feature of the `Nordic Edge AI Lab`_ or one of `ready-to-use models <Nordic Edge AI Lab ready-to-use models_>`_.

.. tabs::

   .. group-tab:: Wakeword detection model

      To replace the model used in wakeword detection stage complete the following steps:

      1. Replace the files inside :file:`src/ww/nrf_edgeai_generated` directory with files downloaded from `Nordic Edge AI Lab`_.
      #. In :c:func:`ww_init`, set the ``ww_model`` pointer using the model getter from the generated files.
      #. Adjust the Kconfig options to tune wakeword detection postprocessing to your model.


   .. group-tab:: Keyword spotting model

      To replace the model used in keyword spotting stage complete the following steps:

      1. Replace the files inside :file:`src/kws/nrf_edgeai_generated` directory with files downloaded from `Nordic Edge AI Lab`_.
      #. In :c:func:`kws_init`, set the ``kws_model`` pointer using the model getter from the generated files.
      #. Update the ``keyword_class`` enumeration and ``keyword_detection_ctxs`` array in the :file:`src/kws/kws.c` file to match keywords spotted by selected model.
      #. Adjust the Kconfig options to tune keyword spotting postprocessing to selected model.

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

This application uses the following Zephyr libraries:

* `Logging`_
* Audio (DMIC)
* UART Driver

API documentation
*****************

.. doxygengroup:: ww_kws
