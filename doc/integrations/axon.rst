.. _axon_integration:

Axon integration
################

.. contents::
   :local:
   :depth: 2

Axon is a neural processing unit (NPU) and software stack that accelerates neural networks and selected signal‑processing algorithms, providing a compiler, driver, simulator, and tools to run workloads either on a Zephyr‑based SoC or on a host‑based simulator.

Integration prerequisites
*************************

* :ref:`setup_sdk`
* Setup of the required Development Kit (DK).

Integration overview
********************

The Axon NPU can accelerate neural networks as well as some feature extraction algorithms. 
In the source code, you can access the driver, sample, simulator, and tool chain files to support Axon NPU. 
Applications can be created to target the SoC (running zephyr) or a software simulator running on the host machine.

The work flows for neural net models and algorithms are different, even though they share many of the same components.

The Axon compiler incorporates the user-supplied model_name into all model-specific symbols, macros, and file names.

Compiled model
==============

The compiled model is placed in a header file of the name :file:`nrf_axon_model_<model_name>_h`.
This file declares the models parameters and compiled code, then encapsulates the model in a ``nrf_axon_nn_compiled_model_s`` instance of name ``nrf_axon_model_<model_name>``.
Structure ``nrf_axon_nn_compiled_model_s`` is declared in :file:`include/nrf_axon_nn_infer.h`.

This structure provides all the model's meta data:

* Model name as text
* Input and output dimensions, locations, quantization parameters
* Pre-allocated output buffer
* Memory

Memory Management
=================

A global, common buffer ``nrf_axon_interlayer_buffer`` is used to store input, intermediate results, and output. This buffer is owned by whichever model is executing.
This buffer must be sized to the requirement of largest need of all the models in the system. Models store their interlayer buffer requirement in ``nrf_axon_model_<model_name>.interlayer_buffer_needed`` and perform a run-time check during model initialization.
Users can see this value declared in the model header file macro ``NRF_AXON_MODEL_<MODEL_NAME>MAX_IL_BUFFER_USED``, then update the allocation in ``NRF_AXON_INTERLAYER_BUFFER_SIZE``.

The synchronous and asynchronous inference APIs accept as parameters the input and output buffers, and will fill/drain the interlayer buffer with input/output in a thread safe manner.

Integration steps
*****************

Complete the following steps:

1. :ref:`Initializing Axon driver <axon_integration_driver_init>`
#. :ref:`Initializing model <axon_integration_init_model>` 
#. :ref:`Executing inference <axon_integration_inference>`
#. :ref:`Integrating the model into an application <axon_integration_model_integration>`

.. _axon_integration_driver_init:

.. rst-class:: numbered-step

Initializing Axon driver
========================

Follow the initialization steps in :ref:`axon_driver_init` to initialize the Axon driver and hardware. 

.. _axon_integration_init_model:

.. rst-class:: numbered-step

Initializing model
==================

The following section describes a compiled Axon model for synchronous or asynchronous inference, and explains how to initialize each type.

Synchronous model inference
---------------------------

Synchronous inference means that the inference call waits for completion before returning.
The synchronous call will wait for the Axon hardware to be available and then claim its exclusive use.

Asynchnronous requests can be made while the Axon is in synchronous mode. 
These requests will be serviced upon exiting of synchronous mode, regardless of any pending synchronous requests.

To initialize the model, invoke the function ``nrf_axon_nn_model_validate(&nrf_model_<model_name>)`` one time at start-up to do basic model validation. 
This will confirm that the global buffers are large enough to handle the model.

Asynchronous model inference
----------------------------

Compiled models need to be initialized prior to asynchronous inference. 
The initialization binds the static, compiled model stored in non-volatile memory (NVM) to a RAM wrapper struct that the driver then manages.
First, you must declare a static (not on the stack) instance of ``nrf_axon_nn_model_async_inference_wrapper_s`` (included in :file:`include/nrf_axon_infer.h`), and then invoke the model by initializing the ``nrf_axon_nn_model_async_init()`` function:

.. code-block:: c

   static nrf_axon_nn_model_async_inference_wrapper_s my_model_wrapper;
   nrf_axon_nn_model_async_init(&my_model_wrapper, nrf_axon_model_<model_name>);

This function also verifies if the interlayer buffer is large enough to accommodate the model's needs.

.. _axon_integration_inference:

.. rst-class:: numbered-step

Executing inference
===================

This section describes the general steps for executing inference using a compiled Axon model.
The same high‑level flow applies to both synchronous and asynchronous inference, with differences in how inference is submitted and how completion is handled.

Follow these steps to execute inference with a compiled Axon model:

#. Ensure the Axon driver has been initialized and the model has been initialized for the selected execution mode.

#. Prepare the input data in packed format::

      input_vector[input_channel_cnt][input_height][input_width]

   The input to populate externally is identified by
   ``nrf_axon_model_<model_name>.external_input_ndx``.

   The input element size is defined by ``inputs[external_input_ndx].byte_width``:

   * ``1`` for ``int8``
   * ``2`` for ``int16``

#. Prepare an output buffer outside of the interlayer buffer, sized to hold the packed output::

      output_buffer[output_channel_cnt][output_height][output_width]

   Output dimension information is available in
   ``nrf_axon_model_<model_name>.output_dimensions``.
   The output rank ordering is channels, height, width.

   The model also declares an internal output buffer in
   ``nrf_axon_model_<model_name>.packed_output_buf``.

   To allocate and use this buffer, define the following macro before including the model header file

   .. code-block:: c

      #define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
      #include "nrf_axon_model_<model_name>_.h"

#. Submit the inference request using the appropriate API for the selected execution mode.

   .. tabs::

      .. tab:: Synchronous inference

         a. Call the following function:
            
            .. code-block:: c

               nrf_axon_nn_model_infer_sync(&nrf_axon_model_<model_name>,
                                           input_vector,
                                           output_buffer)

            The call blocks until inference completes.
            The Axon hardware is reserved exclusively for the duration of the call.

         #. Observe that when when ``nrf_axon_nn_model_infer_sync()`` returns, the ``output_buffer`` is populated with the inference results.

      .. tab:: Asynchronous inference

         a. Submit an inference request using the asynchronous API associated with the initialized wrapper.
            The call returns immediately.
            Execution is scheduled by the driver when the Axon hardware becomes available.

         #. Observe that when the registered completion callback is invoked, the ``output_buffer`` is populated with the inference results.

   In both modes, the driver fills and drains the interlayer buffer in a thread‑safe manner.
   The dimensions of the output tensor are provided in the ``nrf_axon_model_<model_name>.output_dimensions`` field, and the output data is ordered in channels, height, width format.
   This ordering differs from TensorFlow Lite, which uses height, width, channels.

Optional variations
-------------------

The following variations apply to both execution modes:

* ``input_vector`` may be ``NULL`` if input data is written directly to the model input buffer.
* ``output_buffer`` may be ``NULL`` if output data is read directly from the model output buffer.
* Either or both buffers may be ``NULL`` only if there are no other users of the Axon hardware.

When using ``NULL`` buffers, the application is responsible for copying input data into the model and extracting output data from the model.

This approach is useful when the input data is not stored in a contiguous memory block and double buffering would otherwise be required.

To provide input data directly to the model buffer, complete the following steps:

#. Reserve the Axon hardware explicitly:

   .. code-block:: c

      nrf_axon_platform_reserve_for_user();

#. Obtain the model input address:

    .. code-block:: c

      const nrf_axon_nn_compiled_model_input_info_s *model_input =
          nrf_axon_nn_model_1st_external_input(&nrf_axon_model_<model_name>);

#. Copy the input data to the model input address.

#. Invoke the inference API with ``input_vector == NULL``.

Asynchronous inference requests may be submitted while a synchronous inference is executing.
These requests are queued and serviced once the synchronous execution completes.

.. _axon_integration_model_integration:

.. rst-class:: numbered-step

Integrating the model into an application
=========================================

It is recommended to first test your compiled model by using the :ref:`inference test application<include_test_nn_inference>` (:file:`/tests/test_nn_inference`).
Once you have verified that the model works correctly on the test application, you can integrate it into your application.
It is your responsibility to feed data into the model, schedule inference, and respond to the model output.

Ensure you have completed the following:

#. Placed the :file:`nrf_axon_<model_name>_.h` file in the application's include path.

#. Included the header files:

   .. code-block:: c

      #include "nrf_axon_driver.h"
      #include "nrf_axon_inference.h"

#. Included the model header file :file:`nrf_axon_model_<model_name>_.h` in exactly one source file.
   The model symbols are intentionally not declared static, to avoid compiling multiple instances of the model into the application. 
   
#. Updated the following Kconfig values in the application's :file:`prj.conf`` file:

   * Enable the ``NRF_AXON`` Kconfig option.
   * Set ``NRF_AXON_INTERLAYER_BUFFER_SIZE`` to the maximum value needed across all models in the application. 

#. Initialized driver one time at start-up.
#. Initialized the model one-time at start-up for the desired mode of execution, synchronous (``nrf_axon_nn_model_validate``) or asynchronous (``nrf_axon_nn_model_async_init``).
