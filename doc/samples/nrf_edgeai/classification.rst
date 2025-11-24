Classification sample
======================

Parcel State Classification Model
---------------------------------

Description
~~~~~~~~~~~

This sample runs a multi-class classifier that identifies parcel delivery
states (Idle, Shaking, Impact, Free Fall, Carrying, In Car, Placed) from a
stream of acceleration magnitude samples. The model uses a 50-sample input
window and outputs a predicted class and confidence probabilities.

Key details
~~~~~~~~~~~

- Input features: 1 (acceleration magnitude)
- Window size: 50 samples
- Classes: 7 (IDLE, SHAKING, IMPACT, FREE_FALL, CARRYING, IN_CAR, PLACED)

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

Project configuration is provided in ``samples/nrf_edgeai/classification/prj.conf``
and enables the `nrf_edgeai` runtime and common dependencies (newlib, FPU).

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

   west build -b <board> samples/nrf_edgeai/classification

Runtime output (example)
~~~~~~~~~~~~~~~~~~~~~~~~

When successful the sample prints predictions such as:

.. code-block:: console

    In 7 classes, predicted 1 with probability 0.945678
    Expected class SHAKING - predicted SHAKING

and asserts that the predicted class matches the expected label for its
validation cases.

See also
~~~~~~~~

- Source: ``samples/nrf_edgeai/classification/src/main.c``
- Sample metadata: ``samples/nrf_edgeai/classification/sample.yaml``
