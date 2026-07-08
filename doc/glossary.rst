.. _glossary:

Glossary
*********

The glossary defines the key terms used throughout the documentation.

.. glossary::
   :sorted:

   Axon command buffer
      The fundamental unit of work for Axon.
      A command buffer comprises compiled Axon code and a small amount of meta data.

   Axon compiled model
      The Axon NN compiler toolchain produces a header file :file:`nrf_axon_model_<model_name>_.h` that contains the compiled model.
      The compiled model is entirely contained by the :c:type:`nrf_axon_nn_compiled_model_s` structure.

   Inference
      The act of executing the machine learning model.

   Axon interlayer buffer
      A global buffer shared by all Axon models for storing intermediate results.
      It is declared as ``nrf_axon_interlayer_buffer[NRF_AXON_INTERLAYER_BUFFER_SIZE]`` in the platform code.

   Axon intrinsic
      A wrapper function around a small command buffer stored in RAM, allowing certain parameters to be changed at run time.
      It encapsulates a specific piece of functionality.

   Packed buffer
      In the context of Axon, a packed buffer with no gaps between the start of rows in a multi-dimensional shape.
      Axon hardware writes the start of each row to a 32-bit aligned address, so if the row size is not a multiple of 32 bits, a gap appears between rows.
      The compiled model structure describes the shape of the inputs and output.

   Streaming model
      A streaming model uses the TFLite operators VarHandle/AssignVariable/ReadVariable to save/retrieve intermediate results for the next inference.
      The compiled model saves these results to dedicated buffers outside of the interlayer buffer to preserve them between inferences.
