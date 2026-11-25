.. _app_power_measurements:

Power measurements
##################

.. contents::
   :local:
   :depth: 2

This page describes how to measure the power consumption of the :ref:`applications <applications>` during inference, and lists reference inference-window baselines for the nRF7120E and the `nRF54LM20B`_.

Each application can drive a GPIO trace line high while the model runs inference.
A logic analyzer or the digital inputs of the `Power Profiler Kit II (PPK2)`_ record the line together with the current, so you can measure each inference window separately from the rest of the application.

Enabling the trace
******************

The trace is disabled by default.
To enable it, build the application with the :file:`power_measurement.conf` configuration file, using the :makevar:`EXTRA_CONF_FILE` variable.
The file sets the ``CONFIG_INFERENCE_GPIO_TRACE`` Kconfig option.

The application drives the trace line with ``inference_trace_set()`` directly around the inference call.
When the option is disabled, the function is an empty inline function and the trace pins are not configured.

The trace pins are defined in the ``zephyr,user`` node of the board overlay:

.. tabs::

   .. group-tab:: nRF7120 DK

      .. list-table:: Trace pins (nRF7120 DK)
         :widths: auto
         :header-rows: 1

         * - Application
           - Devicetree property
           - GPIO
           - API spanned by the trace
         * - :ref:`Wakeword and Keyword Spotting <app_ww_kws>`
           - ``ww-trace-gpios``
           - **P1.12**
           - ``nrf_edgeai_run_inference()`` for the wakeword model.
         * - :ref:`Wakeword and Keyword Spotting <app_ww_kws>`
           - ``kws-trace-gpios``
           - **P1.13**
           - ``nrf_edgeai_run_inference()`` for the keyword spotting model.
         * - :ref:`Person detection <app_person_detection>`
           - ``infer-trace-gpios``
           - **P1.12**
           - ``nrf_axon_nn_model_infer_sync()`` only (frame capture and decoding run outside the trace).
         * - :ref:`Gesture recognition <app_gesture_recognition>`
           - ``infer-trace-gpios``
           - **P1.12**
           - ``nrf_edgeai_run_inference()`` (CPU feature extraction and Axon model in the same call).

   .. group-tab:: nRF54LM20 DK

      .. list-table:: Trace pins (nRF54LM20 DK)
         :widths: auto
         :header-rows: 1

         * - Application
           - Devicetree property
           - GPIO
           - API spanned by the trace
         * - :ref:`Wakeword and Keyword Spotting <app_ww_kws>`
           - ``ww-trace-gpios``
           - **P1.14**
           - ``nrf_edgeai_run_inference()`` for the wakeword model.
         * - :ref:`Wakeword and Keyword Spotting <app_ww_kws>`
           - ``kws-trace-gpios``
           - **P1.15**
           - ``nrf_edgeai_run_inference()`` for the keyword spotting model.
         * - :ref:`Person detection <app_person_detection>`
           - ``infer-trace-gpios``
           - **P1.14**
           - ``nrf_axon_nn_model_infer_sync()`` only (frame capture and decoding run outside the trace).
         * - :ref:`Gesture recognition <app_gesture_recognition>`
           - ``infer-trace-gpios``
           - **P1.14**
           - ``nrf_edgeai_run_inference()`` (CPU feature extraction and Axon model in the same call).

Hardware setup
**************

Only the SoC supply is measured.
The current of the microphone, camera and IMU is not included.

Depending on the device you are using, connect the PPK2 as follows:

.. tabs::

   .. group-tab:: nRF7120 DK

      Remove the jumper from **P601** first.

      .. list-table:: PPK2 connections (nRF7120 DK)
         :widths: auto
         :header-rows: 1

         * - PPK2
           - Development kit
           - Purpose
         * - VOUT
           - **P601** VBAT
           - Power supply for the SoC.
         * - GND
           - **P601** GND
           - Ground.
         * - Logic port VCC
           - **P1** VIO
           - Sets the logic input threshold to the GPIO voltage.
         * - Logic port GND
           - **P1** GND
           - Ground.
         * - Logic port D0
           - **P1.12**
           - Inference trace.
         * - Logic port D1
           - **P1.13**
           - Keyword spotting trace (Wakeword and Keyword Spotting application only).
         * - Supply voltage
           - 3.6 V
           - Set in the Power Profiler app.

      The measurement point is in front of the nPM1300 regulator, so the result includes regulator losses of roughly 10-20%.

   .. group-tab:: nRF54LM20 DK

      .. list-table:: PPK2 connections (nRF54LM20 DK)
         :widths: auto
         :header-rows: 1

         * - PPK2
           - Development kit
           - Purpose
         * - VOUT
           - **P14** middle pin (VDDM)
           - Power supply for the SoC.
         * - GND
           - **GND**
           - Ground.
         * - Logic port VCC
           - VDDIO
           - Sets the logic input threshold to the GPIO voltage.
         * - Logic port GND
           - **GND**
           - Ground.
         * - Logic port D0
           - **P1.14**
           - Inference trace.
         * - Logic port D1
           - **P1.15**
           - Keyword spotting trace (Wakeword and Keyword Spotting application only).
         * - Supply voltage
           - 3.0 V
           - Set in the Power Profiler app.

      The measurement point is directly at the SoC supply.

Inference trace windows by application
**************************************

The GPIO line is high only during the work described for each application below.
Use these timing patterns when aligning PPK2 digital channels with current pulses.

Wakeword and Keyword Spotting
   ``nrf_edgeai_run_inference()`` is called every 10 ms.
   When not enough audio has been collected for a full model run, it only extracts features and returns ``NRF_EDGEAI_ERR_INPROGRESS``.
   The line is still driven, so the trace shows two short feature-only pulses followed by one full inference pulse every 30 ms.

Person detection
   A 500 ms timer starts one capture cycle: the application captures a frame, then calls ``nrf_axon_nn_model_infer_sync()``.
   Only that call is inside the pulse, so frame capture and output decoding are not included.
   If capture and inference take longer than 500 ms, the next cycle starts immediately.

Gesture recognition
   The IMU is sampled at 100 Hz, and each sample is buffered by ``nrf_edgeai_feed_inputs()``.
   The model uses a 99-sample window that moves forward 33 samples at a time, so ``nrf_edgeai_run_inference()`` runs on every 33rd sample, inside the pulse.
   The trace shows one pulse every 330 ms, covering feature extraction on the CPU and the model on Axon.

.. _app_power_measurements_build:

Building for power measurement
******************************

Build the application from its directory in the :file:`applications` folder.
Use the following commands, where ``<board>`` is ``nrf7120dk/nrf7120e/cpuapp`` or ``nrf54lm20dk/nrf54lm20b/cpuapp``:

.. list-table:: Build commands
   :widths: auto
   :header-rows: 1

   * - Application
     - Configuration
     - Command
   * - Wakeword and Keyword Spotting
     - :file:`prj_release.conf`
     - ``west build -p -b <board> . -- -DFILE_SUFFIX=release -DEXTRA_CONF_FILE=power_measurement.conf``
   * - Person detection
     - :file:`prj.conf`, logging enabled
     - ``west build -p -b <board> . -- -DEXTRA_CONF_FILE=power_measurement.conf``
   * - Gesture recognition
     - :file:`prj_release_no_ble.conf`, logging disabled, MCUboot from the board :file:`sysbuild.conf`
     - ``west build -p -b <board> . -- -DFILE_SUFFIX=release_no_ble -DEXTRA_CONF_FILE=power_measurement.conf -DCONFIG_LOG=n``

The Gesture recognition application keeps its configuration files in :file:`configuration/<board>`, so :file:`power_measurement.conf` is located in that directory.

.. _app_power_measurements_capture:

Capturing SoC current and GPIO traces
*************************************

In the Power Profiler app, record the SoC supply current and the GPIO trace inputs together, so you can bracket each inference window on the current graph.
Complete the following steps after flashing the application build from :ref:`app_power_measurements_build`.

1. Open the Power Profiler app in nRF Connect for Desktop and select the :guilabel:`PPK2` device.
#. Select :guilabel:`Source meter` mode. 
#. Set the supply voltage and enable the power output.
#. Switch the power output off and on again, so the SoC starts without a debugger session.
#. Under :guilabel:`Digital channels`, enable :guilabel:`D0`, and :guilabel:`D1` for the Wakeword and Keyword Spotting application.
#. Set the sampling rate to ``100`` kS/s (100 000 samples per second).
   The inference windows are as short as 0.5 ms, so lower rates blur the edges.
#. Start the capture, let it run for at least 10 seconds, then stop it and save it as a :file:`.ppk2` file.
#. To measure one inference window, zoom in until single pulses are visible and select the graph from the rising edge to the falling edge of a pulse.

Reference inference windows
***************************

This section lists Nordic Semiconductor's reference medians for GPIO-bracketed inference.
After :ref:`app_power_measurements_capture`, compare the duration and charge (energy) of a pulse you selected with the matching row.

.. note::
   The following baselines were captured with the builds in :ref:`app_power_measurements_build`, using the |NCS| ``v3.5.0_preview4``.
   Each application and board was recorded for 60 seconds at 100 kS/s.

The tables use the following columns:

* Window — Which part of the pipeline the row describes (for example, wakeword feature extraction only or full inference).
* Period — Time between comparable pulses, or the application cadence for that row (for example, 10 ms between ``nrf_edgeai_run_inference()`` calls).
* Duration — How long the GPIO trace stays high (median over all pulses in the capture; spread ±0.02 ms).
* Current — Median supply current during the bracketed window.
* Charge (energy) — Charge through the supply during the window; energy is charge multiplied by the tab supply voltage.

.. note::
   The values depend on the silicon revision, the board, and the build.
   Use them as a reference, not as a specification.

.. tabs::

   .. group-tab:: nRF7120E (3.6 V)

      .. list-table:: Inference window (nRF7120E)
         :widths: auto
         :header-rows: 1

         * - Application
           - Window
           - Period
           - Duration
           - Current
           - Charge (energy)
         * - Wakeword and Keyword Spotting
           - Wakeword feature extraction only (2 of every 3 calls)
           - 10 ms
           - 0.50 ms
           - 6.47 mA
           - 3.21 µC (11.6 µJ)
         * - Wakeword and Keyword Spotting
           - Wakeword full inference
           - 30 ms
           - 1.17 ms
           - 6.39 mA
           - 7.46 µC (26.9 µJ)
         * - Wakeword and Keyword Spotting
           - Keyword spotting full inference
           - 30 ms
           - 4.95 ms
           - 6.83 mA
           - 33.8 µC (121.7 µJ)
         * - Person detection
           - Model on Axon
           - 546 ms
           - 18.6 ms
           - 7.58 mA
           - 140.6 µC (506 µJ)
         * - Gesture recognition
           - Feature extraction and model
           - 330 ms
           - 1.25 ms
           - 8.34 mA
           - 10.5 µC (37.7 µJ)

   .. group-tab:: nRF54LM20B (3.0 V)

      .. list-table:: Inference window (nRF54LM20B)
         :widths: auto
         :header-rows: 1

         * - Application
           - Window
           - Period
           - Duration
           - Current
           - Charge (energy)
         * - Wakeword and Keyword Spotting
           - Wakeword feature extraction only (2 of every 3 calls)
           - 10 ms
           - 0.98 ms
           - 3.11 mA
           - 3.06 µC (9.2 µJ)
         * - Wakeword and Keyword Spotting
           - Wakeword full inference
           - 30 ms
           - 2.23 ms
           - 2.70 mA
           - 6.03 µC (18.1 µJ)
         * - Wakeword and Keyword Spotting
           - Keyword spotting full inference
           - 30 ms
           - 9.11 ms
           - 2.71 mA
           - 24.7 µC (74.1 µJ)
         * - Person detection
           - Model on Axon
           - 511 ms
           - 36.1 ms
           - 3.98 mA
           - 143.7 µC (431 µJ)
         * - Gesture recognition
           - Feature extraction and model
           - 330 ms
           - 2.04 ms
           - 3.13 mA
           - 6.38 µC (19.1 µJ)