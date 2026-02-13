.. _supported_operators:

Supported operators
###################

.. contents::
   :local:
   :depth: 2

This section describes the model structure constraints and the set of operators supported by the Axon Compiler.

Model structure
***************

The Axon Compiler supports a wide range of operators commonly used in machine learning models. 
The supported operators include:

* Supporting 8-bit quantized input and output for all layers, with an option to use int32 model output with a configurable radix.
* Supporting stateful behavior between inferences when declared using VarHandle, ReadVariable, or AssignVariable.
* Allowing a maximum of two inputs per node.
* Supporting maximum tensor sizes (height, width, channels) of 1024, 1024, and 512.
* Providing native activation functions: ReLU, ReLU6, and LeakyReLU.
* Providing CPU activation functions: Sigmoid and Tanh.

Operators
*********

The following operators are supported in the current version of Axon Compiler:

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Operator
     - Notes
     - Target
     - Compiler version
   * - Conv1D
     - Max filter width 32, max filter height 16, max stride 31
     - Axon NPU|CPU
     - 1.0.0
   * - Depthwise Conv1D
     - Max filter width 32, max filter height 16, max stride 31
     - Axon NPU|CPU
     - 1.0.0
   * - Conv2D
     - Max filter dimensions 16x16, max stride 31
     - Axon NPU|CPU
     - 1.0.0
   * - Depthwise Conv2D
     - No channel multipliers, max filter dimensions 16x16, max stride 31
     - Axon NPU|CPU
     - 1.0.0
   * - Fully connected
     - Max input length 2048
     - Axon NPU|CPU
     - 1.0.0
   * - Add
     - Vector operation with broadcast on height and/or width
     - Axon NPU|CPU
     - 1.0.0
   * - Multiply
     - Vector operation with broadcast on height and/or width
     - Axon NPU|CPU
     - 1.0.0
   * - Average pooling
     - No padding, max filter dimensions 32x32
     - Axon NPU|CPU
     - 1.0.0
   * - Max pooling
     - Max filter dimensions 32x32
     - Axon NPU|CPU
     - 1.0.0
   * - Mean
     - Includes global average pooling
     - Axon NPU|CPU
     - 1.0.0
   * - Global average pooling
     - Implemented as Mean
     - Axon NPU|CPU
     - 1.0.0
   * - Strided slice
     - Max stride 31
     - Axon NPU|CPU
     - 1.0.0
   * - Concatenate
     - No additional limitations specified
     - Axon NPU|CPU
     - 1.0.0
   * - splitV
     - No additional limitations specified
     - Axon NPU|CPU
     - 1.0.0
   * - Softmax
     - Executed on CPU
     - Axon NPU|CPU
     - 1.0.0

Model recommendations
*********************

TBA
