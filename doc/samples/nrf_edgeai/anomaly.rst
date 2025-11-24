Anomaly detection sample
=========================

Mechanical Gear Anomaly Detection Model
--------------------------------------

Description
~~~~~~~~~~~

This sample runs an anomaly-detection model that monitors dual-axis vibration
data to detect gear faults. The model computes a single anomaly score for a
128-sample window; scores above the configured threshold indicate a fault.

Key details
~~~~~~~~~~~

- Input features: 2 (X and Y vibration axes, interleaved)
- Window size: 128 samples
- Output: single anomaly score (float)
- Threshold: see sample implementation for example threshold value

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

Project configuration is provided in ``samples/nrf_edgeai/anomaly/prj.conf``.

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

   west build -b <board> samples/nrf_edgeai/anomaly

Runtime output (example)
~~~~~~~~~~~~~~~~~~~~~~~~

Typical output includes the anomaly score and a human-readable verdict:

.. code-block:: console

    Anomaly score for GOOD gear data: 0.000010
    Verdict: NORMAL (score < threshold)

.. code-block:: console

    Anomaly score for ANOMALOUS gear data: 0.000120
    Verdict: ANOMALY DETECTED (score >= threshold)

See also
~~~~~~~~

- Source: ``samples/nrf_edgeai/anomaly/src/main.c``
- Sample metadata: ``samples/nrf_edgeai/anomaly/sample.yaml``
