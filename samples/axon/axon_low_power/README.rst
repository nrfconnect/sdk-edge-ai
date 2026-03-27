.. _sample_axon_low_power:

Axon Low Power
##############

.. contents::
   :local:
   :depth: 2

The Axon Low Power sample demonstrates the power efficiency of the Nordic Axon NPU by running a wakeword-class neural network in a continuous sliding-window inference loop.
Between inference sweeps, the CPU and NPU enter a low-power state drawing less than 10 µA of current on average, making the Axon NPU well-suited for Edge AI workloads on battery-powered devices.
Use this sample to profile Axon NPU active and idle current on hardware with a power profiler.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The sample loads the ``okay_nordic`` wakeword model (compiled to :file:`src/generated/nrf_axon_model_okay_nordic.h` by the :ref:`Axon NPU Compiler <axon_npu_tflite_compiler>`) and a set of 102 reference mel-spectrogram frames stored in the :file:`src/mel_test_vector.c` file.
Those frames were captured from a real "okay nordic" utterance using the ``ww_kws`` application.
The model carries approximately 25 KB of int8 weights across 12 layers and maintains 6 persistent recurrent-state buffers, making it representative of a real production wakeword model.

At start-up the sample:

1. Initializes the Axon NPU platform and loads the model.
#. Reads the model's input quantization parameters.
#. Builds 100 quantized input windows by sliding a 3-frame window one frame at a time over the 102 mel frames.
   Each window is packed in channel-first (CHW) order as required by the model: for each mel bin, the three time-step values are stored consecutively.
#. Enters a continuous inference loop that runs all 100 windows per sweep (by calling ``nrf_axon_nn_model_infer_sync`` for each window), then sleeps for ``CONFIG_AXON_LOW_POWER_SLEEP_BETWEEN_SWEEPS_MS`` milliseconds before repeating.

Because inference runs on dedicated NPU silicon, the CPU remains free during neural network inference and can enter an idle state between inferences, resulting in a low overall system duty cycle.
The default build disables all logging, console output, and serial peripherals, giving an accurate view of the NPU's current consumption without UART overhead.
The ``debug`` build variant enables logging so you can observe the inference flow on a serial terminal.

Configuration
*************

|config|

Configuration options
=====================

|sample_kconfig|

.. options-from-kconfig::
   :show-type:

Build types
===========

The sample supports the following build types:

.. list-table:: Axon Low Power build types
   :widths: auto
   :header-rows: 1

   * - Build type
     - File name
     - Description
   * - Default
     - :file:`prj.conf`
     - Logging and console are disabled and Power Management (PM) is enabled.
       Use this build for power measurements.
   * - Debug
     - :file:`prj_debug.conf`
     - Logging is enabled in minimal mode.
       Use this build for functional verification on a serial terminal.
   * - GPIO trace
     - :file:`prj_gpio_trace.conf`
     - Same as the default build, but with ``CONFIG_AXON_LOW_POWER_GPIO_TRACING`` Kconfig option enabled.
       This drives two GPIO pins, which are defined in the board's DeviceTree overlay file (see the :file:`boards/` directory).
       You can capture inference and sweep timing on a logic analyzer or oscilloscope (for example, the `Power Profiler Kit II (PPK2)`_).

See `Custom build types`_ and `Providing CMake options`_ for more information.

Building and running
********************

.. |sample path| replace:: :file:`samples/axon/axon_low_power`

.. include:: /includes/build_and_run.txt

GPIO trace pins
===============

When using the GPIO-trace variant, connect a logic analyzer or oscilloscope to the pins defined in the :file:`boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay` file:

.. list-table::
   :header-rows: 1
   * - Pin (default)
     - Signal
     - Description
   * - P1.04
     - ``infer-trace-gpios``
     - High during each ``nrf_axon_nn_model_infer_sync`` call
   * - P1.05
     - ``sweep-trace-gpios``
     - High for the full sliding-window sweep

Testing
=======

To observe the logging output, build the sample with the ``FILE_SUFFIX=debug`` argument.

|test_sample|

#. |connect_kit|
#. |connect_terminal_kit|
#. Reset the development kit.
#. Observe the logging output in the terminal.

Sample output
-------------

The following output is logged in the terminal when using the ``debug`` build variant:

.. code-block:: console

   I: Model: okay_nordic
   I: Input size: 120
   I: Output size: 1
   I: 102 captured frames -> 100 sliding windows
   I: Axon NPU platform and model ready
   I: Quantization params: zp=-128, mult=1759218604, round=30
   I: Quantized 100 input windows (CHW layout)
   I: Running 100-window sweep (iteration 0)
   I: Sleeping 1000 ms
   I: Running 100-window sweep (iteration 1)
   I: Sleeping 1000 ms
   ...

Dependencies
************

This sample uses the following |EAI| libraries:

* :ref:`Axon NPU driver <axon_driver>`

It uses the following Zephyr libraries:

* `Logging`_
* `Power Management`_
