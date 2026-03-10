.. _supported_operators:

Supported operators
###################

.. contents::
   :local:
   :depth: 2

This section describes the model structure constraints and the set of operators supported by the Axon Compiler.

Model structure
***************

The Axon Compiler has the following constraints on model structure:

* Supports 8-bit quantized input and output for all layers, with an option to use int32 model output with a configurable radix.
* Supports stateful behavior between inferences when declared using VarHandle, ReadVariable, or AssignVariable.
* Allows a maximum of two inputs per node.
* Supports maximum tensor sizes:

  * Height: 1024
  * Width: 1024
  * Channels: 1024

* Provides native activation functions: ReLU, ReLU6, and LeakyReLU.
* Provides CPU activation functions: Sigmoid, Tanh, and Softmax.

.. _supported_operators_reshape:

Memory organization and reshape
*******************************

Axon stores tensors in memory using the following layout:

.. code-block:: c

  int8_t my_axon_tensor[channel_count][height][width];

This differs from TFLite, where channels are the innermost dimension:

.. code-block:: c

  int8_t my_tflite_tensor[height][width][channel_count];

In addition, the Axon hardware aligns the start of each output row to a 32‑bit boundary.
When the tensor width is not a multiple of four, this alignment introduces padding bytes at the end of each row.
Because of these differences, a reshape operation in TFLite, which only changes tensor dimensions without moving data, may require actual data movement on Axon.
In such cases, the reshape is executed on the CPU to perform the required memory reorganization.

Many reshape operations are transparent to Axon and do not require data movement, including:

* Dropping or adding an axis when transitioning between 1D and 2D operators.
* Flattening a multi‑channel output before a fully connected operator.
  The Axon compiler reorders the weights to account for the channel layout.
* Applying a height/width transpose around an operation (for example, rotating the input by 90 degrees, performing the operation, and then rotating the output back).
  In this case, the Axon compiler removes the reshape operations and executes the operation in the unrotated orientation.

.. _supported_operators_list:

Operators
*********

Operators can run entirely on Axon, entirely on the CPU, or be split between the two.
Operators executed on the CPU consume CPU cycles and add a small amount of interrupt latency when they appear in the middle of the model, rather than at the beginning or end.
The following operators are supported in the current version of Axon Compiler:

Convolution operators
=====================

.. list-table::
   :header-rows: 1

   * - Operator
     - Notes / limitations
     - Target
     - Compiler version
   * - Conv1D
     - | Dilation not supported.
       | Implemented as a 2D convolution with either height or width = 1.
       | Max filter width: 32 (when height is 1)
       | Max filter height: 16 (when width is 1)
       | Max stride: 31
     - Axon NPU
     - 1.0.0
   * - Depthwise Conv1D
     - | Max filter width: 32
       | Max filter height: 16
       | Max stride: 31
     - Axon NPU
     - 1.0.0
   * - Conv2D
     - | Dilation supported with width dilation <= 31, and output width 1.
       | Max filter dimensions: 16 x 16
       | Max stride: 31
     - Axon NPU
     - 1.0.0
   * - Depthwise Conv2D
     - | Channel multipliers not supported
       | Dilation not supported
       | Max filter dimensions: 16 x 16
       | Max stride: 31
     - Axon NPU
     - 1.0.0

Fully connected layer
=====================

.. list-table::
   :header-rows: 1

   * - Operator
     - Notes / limitations
     - Target
     - Compiler version
   * - Fully Connected
     - | Maximum input vector length: 2048
       | Maximum number of neurons: 2048
     - Axon NPU
     - 1.0.0

Pooling operators
=================

.. list-table::
   :header-rows: 1

   * - Operator
     - Notes / limitations
     - Target
     - Compiler version
   * - Average Pooling
     - | No padding
       | Max filter dimensions: 32 x 32
     - Axon NPU
     - 1.0.0
   * - Max Pooling
     - | Max filter dimensions: 32 x 32
       | Maximum input/output size: 1024
     - Axon NPU
     - 1.0.0
   * - Mean
     - Includes global average pooling functionality
     - Axon NPU
     - 1.0.0
   * - Global Average Pooling
     - Implemented internally as Mean
     - Axon NPU
     - 1.0.0

Activation functions
====================

.. list-table::
   :header-rows: 1

   * - Operator
     - Notes / limitations
     - Target
     - Compiler version
   * - ReLU
     - Native activation function
     - Axon NPU
     - 1.0.0
   * - ReLU6
     - Native activation function
     - Axon NPU
     - 1.0.0
   * - LeakyReLU
     - Native activation function
     - Axon NPU
     - 1.0.0
   * - Sigmoid
     - Executed on the CPU
     - CPU
     - 1.0.0
   * - Tanh
     - Executed on the CPU
     - CPU
     - 1.0.0
   * - Softmax
     - Partially accelerated on Axon NPU, with CPU assistance
     - Axon NPU | CPU
     - 1.0.0

Elementwise operators
=====================

.. list-table::
   :header-rows: 1

   * - Operator
     - Notes / limitations
     - Target
     - Compiler version
   * - Add
     - Vector operation with broadcast on height and/or width
     - Axon NPU
     - 1.0.0
   * - Multiply
     - Vector operation with broadcast on height and/or width
     - Axon NPU
     - 1.0.0

Tensor manipulation operators
=============================

.. list-table::
   :header-rows: 1

   * - Operator
     - Notes / limitations
     - Target
     - Compiler version
   * - Strided Slice
     - Max stride: 31
     - Axon NPU
     - 1.0.0
   * - Concatenate
     - No additional limitations specified
     - Axon NPU
     - 1.0.0
   * - splitV
     - No additional limitations specified
     - Axon NPU
     - 1.0.0
   * - Reshape
     - | Performed as a CPU operation in those cases where it is not transparent.
       | See :ref:`supported_operators_reshape` for details.
     - CPU
     - 1.0.0

Model design recommendations
****************************

This section provides practical guidance for designing models that compile successfully and run efficiently on the Axon platform.

Preferred architectures
=======================

When designing models for Axon, favor convolution‑based architectures with the following characteristics:

* Using CNN building blocks, such as:

  * Conv1D or Conv2D layers
  * Depthwise separable convolutions without channel multipliers
  * MaxPooling or AveragePooling layers
  * Mean for implementing global average pooling
  * Fully connected layers that stay within supported size limits

* Using activation functions optimized for NPU execution:

  * ReLU
  * ReLU6
  * LeakyReLU

Architectural constraints
=========================

To stay within compiler and hardware limits, ensure that the model structure adheres to the following constraints:

* Limiting tensor dimensions - Do not exceed the operator limits listed in the :ref:`supported_operators_list` section.

* Limiting fully connected layers:

  * Input vector size to 2048 or less
  * Number of neurons to 2048 or less

* Limiting graph complexity:

  * Using no more than two inputs per node
  * Avoiding complex reshape patterns between convolutional and dense layers

* Avoiding unsupported convolution patterns:

  * Dilation, unless the output dimension is 1 x 1
  * Depthwise convolution with dilation

Quantization guidance
=====================

To achieve optimal performance and predictable accuracy, train models using 8‑bit quantization awareness when targeting deployment on Axon.

Deployment strategy
===================

For time‑series and embedded use cases, the following architectural patterns are recommended:

* Typical Conv1D‑based pipelines:

  * Conv1D → ReLU → Pooling → Conv1D → ReLU → Global Average (Mean) → Dense

* Depthwise‑separable pipelines:

  * Depthwise Conv → Pointwise Conv → ReLU → Pooling → Dense

* Fully convolutional models:

  * Convolutional layers followed by final Mean aggregation

When replacing recurrent or attention‑based models, consider convolutional alternatives such as:

* Temporal convolutional networks without dilation
* Stacked Conv1D blocks combined with pooling and global averaging

These patterns maximize compatibility, performance, and compilation success on the Axon platform.
