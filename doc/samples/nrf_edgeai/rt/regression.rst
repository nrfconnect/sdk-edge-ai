Regression sample
=================

Air Quality Prediction Regression Model
---------------------------------------

Description
~~~~~~~~~~~

This sample runs a generated regression model that predicts a continuous air
quality value from gas sensor readings and environmental measurements. The
sample performs validation across a 29-sample test dataset and prints the
predicted value, expected value and absolute error for each case.

Key details
~~~~~~~~~~~

- Input features: 9 (CO, 5x PT08S sensors, Temperature, RH, AH)
- Window size: 1 (inference after each sample)
- Outputs: single floating-point prediction

Supported boards
~~~~~~~~~~~~~~~~

The example is exercised for the following board targets (use with
``west build -b``):

- ``nrf52dk/nrf52832``
- ``nrf52840dk/nrf52840``
- ``nrf5340dk/nrf5340/cpuapp``
- ``nrf5340dk/nrf5340/cpuapp/ns``
- ``nrf54l15dk/nrf54l05/cpuapp``
- ``nrf54l15dk/nrf54l10/cpuapp``
- ``nrf54l15dk/nrf54l15/cpuapp``
- ``nrf54h20dk/nrf54h20/cpuapp``
- ``thingy53/nrf5340/cpuapp``

Configuration
~~~~~~~~~~~~~

Project configuration is provided in ``samples/nrf_edgeai/regression/prj.conf``.

.. code-block:: ini

   CONFIG_NRF_EDGEAI=y
   CONFIG_NEWLIB_LIBC=y
   CONFIG_FPU=y
   CONFIG_CONSOLE=y
   CONFIG_UART_CONSOLE=y
   CONFIG_RTT_CONSOLE=n
   CONFIG_NEWLIB_LIBC_FLOAT_PRINTF=y

Build
~~~~~

From the repository root:

.. code-block:: console

   west build -b <board> samples/nrf_edgeai/regression

Runtime output (example)
~~~~~~~~~~~~~~~~~~~~~~~~

The sample prints per-sample validation results like:

.. code-block:: console

    Air quality - Predicted value: 12.345678, Expected value: 14.300000, absolute error 1.954322


See also
~~~~~~~~

- Source: ``samples/nrf_edgeai/regression/src/main.c``
- Sample metadata: ``samples/nrf_edgeai/regression/sample.yaml``
