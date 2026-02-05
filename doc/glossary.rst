.. _glossary:

Glossary
*********

command_buffer
--------------

The fundamental unit of work for Axon is a command buffer. A command buffer comprises compiled Axon code and a small amount of meta data.

compiled model
--------------

The Axon NN compiler tool chain produces a header file :file:`nrf_axon_model_<model_name>_.h` that contains the compiled model. 
The compiled model is entirely contained by an nrf_axon_nn_compiled_model_s structure. This structure is declared in :file:`include/nrf_axon_nn_infer.h`.

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
