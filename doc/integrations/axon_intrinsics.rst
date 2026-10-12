.. _ug_axon_dsp_intrinsics_integration:

Axon DSP intrinsics integration
###############################

.. contents::
   :local:
   :depth: 2

This guide explains how to integrate Axon NPU DSP intrinsic functions into your application.
DSP intrinsics are pre-defined functions that run signal-processing workloads directly on the Axon NPU through the Axon driver.

Integration prerequisites
*************************

* :ref:`setup_sdk`
* A development kit with `Axon NPU`_ hardware, or the Axon simulator for host-based testing.
* A successful run of :ref:`test_axon_intrinsics`, as described in :ref:`quick_start_axon_dsp_intrinsics`.
  Use that test to confirm your environment before integrating intrinsics into your own application.

Integration overview
********************

DSP intrinsics use the same Axon driver and platform initialization flow as compiled model inference, but they do not require a compiled model, interlayer buffer sizing for a network, or the Axon TFLite compiler.

The API is documented in :c:group:`nrf_axon_dsp_intrinsics`.
See :ref:`supported_dsp_intrinsics` for the full list of functions.

Integration steps
*****************

Complete the following steps:

1. :ref:`axon_dsp_intrinsics_driver_init`
#. :ref:`axon_dsp_intrinsics_call_functions`
#. :ref:`axon_dsp_intrinsics_app_integration`

.. _axon_dsp_intrinsics_driver_init:

.. rst-class:: numbered-step

Initializing the driver
=======================

Initialize the Axon driver before calling any DSP intrinsic function:

#. Call the platform initialization function:

   .. code-block:: c

      nrf_axon_platform_init();

   On Zephyr, the Axon base address is obtained from the device tree during platform initialization.

   .. note::

      Do not create or manage a driver handle.
      Axon is implemented as a singleton, and the driver serializes access internally.

.. _axon_dsp_intrinsics_call_functions:

.. rst-class:: numbered-step

Calling intrinsic functions
===========================

Include the DSP intrinsics header and call the function that matches your workload:

.. code-block:: c

   #include "drivers/axon/nrf_axon_driver.h"
   #include "drivers/axon/nrf_axon_dsp_intrinsics.h"

Most intrinsic functions accept input and output buffers, workload parameters, a synchronous blocking mode, and a ``keep_reservation`` flag.
Set ``keep_reservation`` to ``true`` when you plan to run another Axon operation immediately afterward.

For short standalone workloads, use :c:enumerator:`NRF_AXON_SYNC_MODE_BLOCKING_POLLING` as the blocking mode.

Refer to :ref:`supported_dsp_intrinsics` for function-specific parameter requirements and data type conventions.

.. _axon_dsp_intrinsics_app_integration:

.. rst-class:: numbered-step

Integrating intrinsics into your application
============================================

Wire the driver initialization and intrinsic calls into your application startup and processing flow.

Ensure you have completed the following:

#. Enabled the ``CONFIG_NRF_AXON`` Kconfig option in your application's :file:`prj.conf` file.
#. Called :c:func:`nrf_axon_platform_init` once at startup before any intrinsic function.
#. Included :file:`nrf_axon_driver.h` and :file:`nrf_axon_dsp_intrinsics.h` where intrinsic functions are used.
#. Validated the intrinsic outputs your application depends on, using known-good input vectors or reference data for your workload.

When combining intrinsics with compiled model inference in the same application, ensure that Axon hardware access remains serialized through the driver and size the interlayer buffer for your largest model as described in :ref:`ug_axon_inference_integration`.

.. note::

   :ref:`test_axon_intrinsics` is a fixed regression test for the SDK.
   It confirms that the platform and driver work, but it does not validate your application's buffers, parameters, or data flow.
   If intrinsic calls fail unexpectedly in your application, re-run :ref:`test_axon_intrinsics` to rule out environment or toolchain issues before debugging application logic.
