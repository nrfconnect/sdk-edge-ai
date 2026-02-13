.. _axon_driver:

Axon driver
###########

.. contents::
   :local:
   :depth: 2

The following page describes how to initialize the Axon driver and prepare the Axon NPU for running inference workloads.

Overview
********

The Axon NPU is a peripheral processor that runs independently of the CPU.
Use the Axon driver to control the Axon hardware and run inference workloads. 
The driver provides the following functionality:

* Initializing the Axon hardware and driver.
* Submitting jobs to the hardware and handling hardware events.
* Running inference in synchronous or asynchronous mode.
* Executing intrinsics, which are small, pre-compiled Axon code snippets that perform limited functions.

The driver also includes wrapper and test functions for managing compiled AI models. 
These are provided in source form.
The Axon driver is implemented as a platform-independent library. 
It does not depend directly on the host platform. 
All platform-specific behavior is handled by the nRF Axon platform component.
This separation allows the same driver library to be built and used on Zephyr, in a simulator, or in bare-metal environments.

.. _axon_driver_init:

Initializing driver
*******************

Follow these steps to initialize the Axon driver:

1. Call the platform initialization function

   .. code-block:: console 

      nrf_axon_platform_init()

   This function is platform-specific, but you must provide the Axon base address (``nrf_axon_driver_init(base_address``).
   You can obtain ``base_address`` from the device tree on Zephyr.

   During initialization, the driver powers on Axon by calling the ``nrf_axon_platform_vote_for_power()`` function.
   The driver then verifies that Axon exists at the specified base address. 
   Axon remains powered on after initialization.

   .. note::
      
      Do not create or manage a driver handle.
      Axon is implemented as a singleton, and the driver serializes access internally.

#. Before starting a new inference session on a streaming-style model (where intermediate results are fed forward), initialize the model’s persistent variables:

   .. code-block:: console 

      nrf_axon_nn_model_init_vars(&my_model_wrapper);

   This sets all persistent variables to their quantized zero-point values.

#. Refer to further instructions on :ref:`integrating the driver into your application <ug_axon_integration>`.
