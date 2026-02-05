.. _axon_driver:

Axon drivers
############

.. contents::
   :local:
   :depth: 2

Overview
********

Axon NPU is physically implemented as a peripheral processor that executes indepedently of the CPU.

The Axon driver contains the following core functionality:
* Initialize the driver and hardware.
* Submit jobs to the hardware and respond to hardware events. Synchronous and Asynchronous modes are supported.
* Execute 
* Implement intrinsics; snippets of pre-compiled axon code that perform limited functions.

Additionally, the Axon driver provides wrapper and test functions for managing compiled AI models. This functionality is provided in source form.

The Axon driver is implemented as library code that has no direct dependencies on the host platform. Platform interactions are implemented separately in the nRF Axon platform component.

The platform abstraction allows the same driver library to be compiled for Zephyr, simulator, and bare-metal targets.

.. _axon_driver_init:

Driver Initialization
*********************

The axon driver is initialized indirectly via the platform function ``nrf_axon_platform_init()``.

This is implemented differently for different platforms, but this function is required to call::

   nrf_axon_driver_init(base_address) 

where base_address is obtained from the device tree on Zephyr. This function will power-on axon indirectly through::

   nrf_axon_platform_vote_for_power()
   
and verify the existence of axon at base_address. Axon remains in the powered-on state.

Note that there is no driver handle associated with Axon. Axon is instantiated as a singleton, and access to it is serialized by the driver, so there is no need for a handle.

Prior to starting a new inference session on a streaming style model (ie, past intermediate results are fed-foward), invoke::

   nrf_axon_nn_model_init_vars(&my_model_wrapper); 
   
to initialize the persistent model variables to their quantized zero-point values.

