
.. _nrf_axon_4root:

Axon Driver
###############################################

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

The Axon driver is implemented as library code that has no direct dependencies on the host platform. Platform interactions are implemented separately in the `<nrf_axon_platform>`_ component.

The platform abstraction allows the same driver library to be compiled for Zephyr, simulator, and bare-metal targets.

Terminology
*******************

command_buffer
--------------

The fundamental unit of work for Axon is a command buffer. A command buffer comprises compiled Axon code and a small amount of meta data.

compiled model
--------------

The Axon NN compiler tool chain produces a header file nrf_axon_model_<model_name>_.h that contains the compiled model. 
The compiled model is entirely contained by an nrf_axon_nn_compiled_model_s structure. This structure is declared in `<include/nrf_axon_nn_infer.h>`_.

inference
---------

The act of executing the compiled model.

interlayer buffer
-----------------

The interlayer buffer is a global buffer shared by all models for storing intermediate results.
It is declared as nrf_axon_interlayer_buffer[NRF_AXON_INTERLAYER_BUFFER_SIZE] in the platform code. 

intrinsic
---------

An intrinsic is a wrapper function around a small command buffer that is stored in RAM so that certain parameters can be changed at run time. The intrinsic encapsulates specific functionality.

packed buffer
---------------

In the context of Axon, a packed buffer is one where there are no gaps between the start of
rows in a multi-dimensional shape. Axon hardware writes the start of each row to a 32bit aligned address.
If the row size is not a multiple of 32bits, there is a gap between the rows. The compiled model structure describes the shape of the inputs and output.

streaming model
---------------

A streaming model uses the TFLite operators VarHandle/AssignVariable/ReadVariable to save/retrieve intermediate results for the next inference. The compiled model will save these results to dedicated buffers outside of the interlayer buffer so that they are preserved between inferences.

Driver Usage
************

Driver Initialization
---------------------

The axon driver is initialized indirectly via the platform function
::
   nrf_axon_plaform_init()

This is implemented differently for different platforms, but this function is required to call
::
   nrf_axon_driver_init(base_address) 

where base_address is obtained from the device tree on Zephyr. This function will power-on axon indirectly through 
::
   nrf_axon_platform_vote_for_power()
   
and verify the existence of axon at base_address. Axon remains in the powered-on state.

Note that there is no driver handle associated with Axon. Axon is instantiated as a singleton, and access to it is serialized by the driver, so there is no need for a handle.

Prior to starting a new inference session on a streaming style model (ie, past intermediate results are fed-foward), invoke 
::
   nrf_axon_nn_model_init_vars(&my_model_wrapper); 
   
to initialize the persistent model variables to their quantized zero-point values.

Working with the compiled model
-------------------------------
The Axon compiler incorporates the user-supplied model_name into all model-specific symbols, macros, and file names.

The compiled model is placed in a header file of the name nrf_axon_model_<model_name>_h.
This file declares the models parameters and compiled code, then encapsulates the model in a ``nrf_axon_nn_compiled_model_s`` instance of name ``nrf_axon_model_<model_name>``.
Structure ``nrf_axon_nn_compiled_model_s`` is declared in `<include/nrf_axon_nn_infer.h>`_.
This structure provides all the model meta data;
* model name as text.
* input/output dimensions,locations, and quantization parameters.
* pre-allocated output buffer.
* Memory needs.
* etc.

Memory Management
-----------------
A global, common buffer ``nrf_axon_interlayer_buffer`` is used to store input, intermediate results, and output. This buffer is owned by whichever model is executing.
This buffer must be sized to the requirement of largest need of all the models in the system. Models store their interlayer buffer requirement in ``nrf_axon_model_<model_name>.interlayer_buffer_needed`` and perform a run-time check during model initialization.
Users can see this value declared in the model header file macro ``NRF_AXON_MODEL_<MODEL_NAME>MAX_IL_BUFFER_USED``, then update the allocation in ``NRF_AXON_INTERLAYER_BUFFER_SIZE``.

The synchronous and asynchronous inference APIs accept as parameters the input and output buffers, and will fill/drain the interlayer buffer with input/output in a thread safe manner.

Synchronous Model Inference
---------------------------
Synchronous inference simply means that the inference call waits for completion before returning.
The synchronous call will wait for the Axon hardware to be available then claim its exclusive use.
Once completed, Axon is available to other users.

Asynchnronous requests can be made while the Axon is in synchronous mode. 
These requests will be serviced upon exiting of synchronous mode, regardless of any pending synchronous requests.



Model Initialization
================================
Invoke the function `nrf_axon_nn_model_validate(&nrf_model_<model_name>)`
one time at start-up to do basic model validation. This will confirm that the global buffers are large enough to handle the model.


Inference
================================
Invoke the  function
::
   nrf_axon_nn_model_infer_sync(&nrf_model_<model_name>, input_vector, output_buffer)
to perform a synchronous inference.

input_vector points to the packed input in the format
::
   input_vector[input_channel_cnt][input_height][input_width]
The field
::
   nrf_model_<model_name>.external_input_ndx
informs which of the inputs is to be populated externally (as opposed to being maintained internally by the model).
The input data type will generally be int8_t
::
   inputs[external_input_ndx].byte_width 
specifies the data type. (1=>int8, 2=>int16)

output_buffer points to a buffer outside of the interlayer buffer sized to hold the packed output. It is also in the format:
::
   output_buffer[output_channel_cnt][output_height][output_width]
The model has a declared buffer sized for the output in the field:
::
   nrf_model_<model_name>.packed_output_buf
To access it, define this macro prior to including the model header file:
::
   #define NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER 1
   #include "nrf_axon_model_<model_name>_.h"


Either/both input_vector and output_buffer can be NULL, but this is only safe if there are no other users of Axon. The user will have 
to copy the input_vector to the model's input pointer, and extract the output from the model's output pointer.

If an input_vector is not readily available (due to the input not being in a continous memory block), double buffering of the input can be avoided by mirroring the technique in nrf_axon_nn_model_infer_sync().
1. Reserve the Axon hardware directly:
::
   nrf_axon_platform_reserve_for_user();
2. Get the model's input address:
::
   const nrf_axon_nn_compiled_model_input_info_s *model_input = nrf_axon_nn_model_1st_external_input(&nrf_axon_model_<model_name>);
3. Copy the input data to the model's input address.
4. Invoke nrf_axon_nn_model_infer_sync() with input_vector==NULL.


Inference results
=================

Upon returning from 
::
   nrf_axon_nn_model_infer_sync(&nrf_model_<model_name>, input_vector, output_buffer)
output_buffer will be populated with the model results. Users can process the results directly from this buffer.
The dimensions of the output are in the field
::
   nrf_axon_model_<model_name>.output_dimensions
The rank of multiple dimension output is channels=>height=>width. 
Note that this is different from TFLite ranking which is height=>width=>channels.


Asynchronous Model Inference
----------------------------

Model Initialization
====================

Compiled models need to be initialized prior to asynchronous inference. The initialization binds the static, compiled model stored in NVM to a RAM wrapper struct that the driver then manages.
The user declares a static (ie, not on the stack) instance of nrf_axon_nn_model_async_inference_wrapper_s (declared in `<include/nrf_axon_infer.h>`), then invokes nrf_axon_nn_model_async_init().
::
   static nrf_axon_nn_model_async_inference_wrapper_s my_model_wrapper;
   nrf_axon_nn_model_async_init(&my_model_wrapper, nrf_axon_model_<model_name>);

This function also verifies that the interlayer buffer is large enough to accommodate the model's needs.



Model Integration
*****************
It is recommended that users first test their compiled model by using the test application `</samples/test_nn_inference>`_.
Once the model is verified to work on the test application, it can be integrated into the user's application.
It is the user's responsibility to feed data into the model, schedule inference, and respond to the model output.

Set-up
------
1. Place the 
::
   nrf_axon_<model_name>_.h 
file in the application's include path.
2. Include the header files
::
   #include "nrf_axon_driver.h"
   #include "nrf_axon_inference.h"
3. Include the model header file 
::
   nrf_axon_model_<model_name>_.h
in exactly one source file. The contents are not declared static intentionlly to prevent duplicates of the model from being compiled into the application.
4. Update the kconfig values:
::
   NRF_AXON='y'
   NRF_AXON_INTERLAYER_BUFFER_SIZE=<maximum value needed across all models in the application>
5. Perform driver initialization one time at start-up.
6. Initialize the model one-time at start-up for the desired mode of execution, synchronous (``nrf_axon_nn_model_validate``) or asynchronous (``nrf_axon_nn_model_async_init``).

Inference
---------
Follow the sections above for performing inference in either synchronous or asynchronous model.

