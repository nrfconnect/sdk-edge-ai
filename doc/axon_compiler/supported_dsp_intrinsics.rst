.. _supported_dsp_intrinsics:

Axon DSP intrinsics
###################

.. contents::
   :local:
   :depth: 2

Axon intrinsics are functions that use the Axon NPU to perform DSP-like and DMA vector operations using fixed-point arithmetic.
They are declared in the :c:group:`nrf_axon_dsp_intrinsics` API and run directly on the NPU without compiling a TensorFlow Lite model.

Fixed-point arithmetic
**********************

All Axon intrinsics use fixed-point arithmetic.
Results saturate at the maximum and minimum limits of the output size.
The ``rounding_bits`` parameter sets how many bits to round off each result.

A variety of input and output bit widths are supported:

* Inputs can be 8, 16, 24, and in rare cases 32 bits.
* Outputs can be 8, 16, 24, and 32 bits.
* 24-bit input and output values are sign-extended to 32 bits internally.

Each intrinsic targets a specific combination of input and output bit widths.
The intrinsic function name incorporates its input and output bit widths.

Not all permutations of bit widths are implemented for every operation.

Synchronous execution
*********************

Axon intrinsics execute synchronously.
Each intrinsic first acquires a hardware-access mutex.
This step is short-circuited if the calling thread already owns the mutex.

You can select the blocking mechanism, event or polling, through the ``block_mode`` parameter.
Polling is recommended in most cases unless there is another thread that occupies the CPU while the intrinsic executes.
Intrinsic execution times are often too short for the power saved during CPU sleep to exceed the overhead of entering and exiting sleep.

Upon completion, the hardware-access mutex is released if ``keep_reservation`` is ``false``.
Acquiring the mutex adds overhead.
Set ``keep_reservation`` to ``true`` to retain the mutex across a sequence of intrinsics, then set it to ``false`` on the final call in the sequence.

Performance benefits compared to the CPU
*****************************************

Certain Axon intrinsics outperform the CPU.
Exponential, natural log, FFT, square root, and FIR filters almost always perform significantly better than Cortex-M33 execution.

Other intrinsics offer a smaller improvement that becomes noticeable only with longer vectors.
Examples include ``a*x + b*y``, accumulation, and MAR (dot product) operations.

Developing with Axon intrinsics
*******************************

Develop algorithms with the Axon simulator before testing and deploying on target hardware.
Applications under :file:`tests/axon` can serve as templates for builds targeting both the simulator and Zephyr.

:ref:`test_axon_intrinsics` invokes each supported intrinsic at least once.
See :ref:`quick_start_axon_dsp_intrinsics` for build and run instructions.

Intrinsic calling conventions
*****************************

All intrinsics use fixed-point arithmetic with configurable rounding and saturation to the output bit-width maximum and minimum values.

* ``rounding_bits`` specifies the rounding amount.
  It is functionally equivalent to a right shift followed by rounding.
* The intrinsic name incorporates input and output bit widths (8, 16, 24, and 32).
  A single number implies the same input and output bit width.
  Two numbers indicate input and output bit widths respectively.
  Three numbers indicate the first input, second input, and output bit widths.
* 24-bit input must be sign-extended to 32 bits.
  24-bit output is sign-extended to 32 bits.
* Some intrinsics can generate 32-bit output, but very few consume 32-bit input.
  32-bit output typically must be consumed by the CPU directly.
* Intrinsics reserve hardware access through a driver-managed mutex.
  Set ``keep_reservation`` to ``true`` to avoid releasing and re-acquiring hardware access between consecutive intrinsic calls.
  Set it to ``false`` on the last intrinsic in a sequence.
* Some intrinsics are compound operations that execute multiple hardware commands.
  Rounding may be applied at one or more stages.
  Refer to the :c:group:`nrf_axon_dsp_intrinsics` API documentation for details.

Intrinsic listing
*****************

The following intrinsics are currently supported.
The tables below summarize the available functions.
Click a function name to open its API reference.

FFT
===

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - :c:func:`nrf_axon_fft_24`
     - 24-bit complex FFT on unpacked 32-bit samples.
   * - :c:func:`nrf_axon_fft_power_24`
     - 24-bit complex FFT followed by power spectrum computation.

FIR filters
===========

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - :c:func:`nrf_axon_fir_24_24_24`
     - FIR filter with 24-bit input, 24-bit coefficients, and 24-bit output.
   * - :c:func:`nrf_axon_fir_24_16_24`
     - FIR filter with 24-bit input, 16-bit coefficients, and 24-bit output.
   * - :c:func:`nrf_axon_fir_2d_16_16_32_decimate`
     - 2D FIR filter with decimation.
   * - :c:func:`nrf_axon_fir_2d_16_16_24_decimate`
     - 2D FIR filter with 24-bit output and decimation.
   * - :c:func:`nrf_axon_fir_16_16_32_decimate`
     - 1D FIR filter with decimation.
   * - :c:func:`nrf_axon_fir_16_16_32_1024_256_decimate_1`
     - 1D FIR filter with decimation factor 1 for 1024-input, 256-tap workloads.
   * - :c:func:`nrf_axon_fir_16_16_32_1024_256_decimate_4`
     - 1D FIR filter with decimation factor 4 for 1024-input, 256-tap workloads.
   * - :c:func:`nrf_axon_fir_cplx_2d_16_16_24_decimate`
     - Complex 2D FIR filter with decimation.
   * - :c:func:`nrf_axon_fir_cplx_2d_16_16_32_decimate`
     - Complex 2D FIR filter with 32-bit output and decimation.

Vector operations
=================

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - :c:func:`nrf_axon_xty_24_24_24`
     - Element-wise multiply of two 24-bit vectors.
   * - :c:func:`nrf_axon_xty_16_16_32`
     - Element-wise multiply of two 16-bit vectors with 32-bit output.
   * - :c:func:`nrf_axon_xty_16_16_32_output_stride`
     - Element-wise multiply with configurable output stride and rounding.
   * - :c:func:`nrf_axon_xpy_24_24_24`
     - Element-wise addition of two 24-bit vectors.
   * - :c:func:`nrf_axon_xmy_24_24_24`
     - Element-wise subtraction of two 24-bit vectors.
   * - :c:func:`nrf_axon_xspys_24_24_24`
     - Multiply a vector by a scalar value.
   * - :c:func:`nrf_axon_xspys_24_24_24_input_stride2`
     - Scalar multiply with input stride of 2.
   * - :c:func:`nrf_axon_xsmys_24_24_24`
     - Multiply a vector by a scalar, then subtract another vector.
   * - :c:func:`nrf_axon_axpby_24_24_24`
     - Affine combination ``a*x + b*y`` of two vectors.
   * - :c:func:`nrf_axon_axpb_24_24`
     - Affine transform ``a*x + b`` on a vector.
   * - :c:func:`nrf_axon_axpb_2d_8_16`
     - 2D affine transform on 8-bit input with 16-bit output.
   * - :c:func:`nrf_axon_axpb_2d_16_16`
     - 2D affine transform on 16-bit data.
   * - :c:func:`nrf_axon_xs_24_24`
     - Scale a vector by a scalar value.
   * - :c:func:`nrf_axon_abs_24_24_24`
     - Element-wise absolute value.

Math functions
==============

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - :c:func:`nrf_axon_sqrt_24`
     - Square root on 24-bit fixed-point values.
   * - :c:func:`nrf_axon_logn_11p12`
     - Natural logarithm with 11.12 fixed-point format.
   * - :c:func:`nrf_axon_exp_11p12`
     - Exponential function with 11.12 fixed-point format.

Matrix and accumulation
=======================

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - :c:func:`nrf_axon_mar_16_24_32`
     - Matrix multiply with 16-bit and 24-bit operands and 32-bit output.
   * - :c:func:`nrf_axon_mar_16_24_24`
     - Matrix multiply with 24-bit output.
   * - :c:func:`nrf_axon_mar_16_16_32`
     - Matrix multiply with 16-bit operands and 32-bit output.
   * - :c:func:`nrf_axon_mar_24_24_32`
     - Matrix multiply with 24-bit operands and 32-bit output.
   * - :c:func:`nrf_axon_mar_24_24_24`
     - Matrix multiply with 24-bit operands and output.
   * - :c:func:`nrf_axon_mar_16_16_24`
     - Matrix multiply with 16-bit operands and 24-bit output.
   * - :c:func:`nrf_axon_marx_24_32`
     - Matrix multiply with transposed right-hand operand.
   * - :c:func:`nrf_axon_marx_16_32`
     - Matrix multiply with 16-bit input and transposed right-hand operand.
   * - :c:func:`nrf_axon_marx_24_24`
     - Matrix multiply with transposed right-hand operand and 24-bit output.
   * - :c:func:`nrf_axon_marx_16_24`
     - Matrix multiply with 16-bit input, transposed right-hand operand, and 24-bit output.
   * - :c:func:`nrf_axon_acc_16_24`
     - Accumulate vector elements into a scalar.
   * - :c:func:`nrf_axon_acc_16_32`
     - Accumulate 16-bit vector elements into a 32-bit scalar.
   * - :c:func:`nrf_axon_acc_24_32`
     - Accumulate 24-bit vector elements into a 32-bit scalar.
   * - :c:func:`nrf_axon_acc_24_24`
     - Accumulate 24-bit vector elements into a 24-bit scalar.
   * - :c:func:`nrf_axon_l2norm_16_24`
     - L2 norm of a 16-bit vector with 24-bit output.
   * - :c:func:`nrf_axon_l2norm_16_32`
     - L2 norm of a 16-bit vector with 32-bit output.
   * - :c:func:`nrf_axon_l2norm_24_24`
     - L2 norm of a 24-bit vector.
   * - :c:func:`nrf_axon_l2norm_24_32`
     - L2 norm of a 24-bit vector with 32-bit output.

Memory and utility
==================

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - :c:func:`nrf_axon_memset_32_output_stride`
     - Fill a buffer with a 32-bit value using configurable output stride.
   * - :c:func:`nrf_axon_saturate_32_24`
     - Saturate 32-bit values to 24-bit range.
   * - :c:func:`nrf_axon_saturate_32_8`
     - Saturate 32-bit values to 8-bit range.

2D operations
=============

.. list-table::
   :header-rows: 1

   * - Function
     - Description
   * - :c:func:`nrf_axon_dma_2d`
     - 2D memory copy with configurable width, height, and strides.
