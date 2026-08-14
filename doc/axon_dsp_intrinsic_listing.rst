.. _axon_dsp_intrinsics_listing:

DSP Intrinsics
##############

.. contents::
   :local:
   :depth: 2


Axon intrinsics are functions that utilize Axon NPU to perform a variety of DSP-like and DMA vector operations using fixed-point arithmetic.
They are declared in the header file
::
   include/drivers/axon/nrf_axon_dsp_intrinsics.h

Fixed-point Arithmetic
**********************
All Axon intrinsics are fixed point. Results will saturate at the maximum and minimum limits of the output size.

Users should be comfortable with the fixed-point arithmetic concept managing the radix. All Axon intrinsics allow the caller to specify the number of bits to round the result by via the `rounding_bits parameter.

A variety of input and output data bit-widths are supported. Inputs can be 8, 16, 24 (and in rare  occaisions, 32) bits.
Outputs can be 8, 16, 24, and 32 bits. (24bit input and output is sign extended to 32bit.)

Each intrinsic is for a specific combination of input and output bit-widths. The intrinsic function name will incorporate its input/output bitwidths.

Note that not all permutations of bit-widths have been implemented for all operations.

Synchronous Execution
*********************

Axon intrinsics execute synchronously.

Each intrinsic first acquires a hardware-access mutex (this is short-circuited if the thread already owns the hardware mutex).

The user selects the blocking mechanism (event or polling) via the ``block_mode`` parameter, but it is recommended to always use polling unless it is known that another thread will occupy the cpu while the intrinsic executes.
This is because the intrisic execution time is too short to make the power consumption saved while in CPU sleep exceed the power consumption overhead of the CPU entering and exiting sleep.

Upon completion, the hardware-access mutex is released if the parameter ``keep_reservation`` is false. There is some overhead involved in acquiring the mutex. 
The ``keep_reservation`` parameter allows the mutex to be acquired and retained for a sequence of intrinsics before it is released on the last one.

Performance Benefits Compared to the CPU
****************************************

Certain Axon intrinsics out-perform the CPU.

Exp(), Natural Log, FFT, square-root, and FIRs will almost always significantly outperform the Cortex M33.


Other intrinsics have a marginal improvement that requires longer vectors to realize.

aX + bY, Accum, MAR (dot product) fall into this category.

Developing with Axon Intrinsics
*******************************

It is recommended that users develop their algorithms using the Axon Simulator before testing and deploying to target.
The applications under ``tests/axon`` can be used as templates for creating applications that can be compiled for the simulator and for Zephyr targets.

``tests/axon/intrinsics`` invokes each of the intrinsics at least once.

Intrinsic Calling Conventions
*****************************

All intrinsics are fixed point with configurable rounding and saturation to the output bitwidth maximum and minimum values.

* Parameter ``rounding_bits`` specifies the rounding amount. It is functionally equivalent to a right shift followed by rounding.
* The name of the intrinsic will incorporate its input and ouput bitwidths (8, 16, 24, and 32). A single number implies that the input and output bitwidths are the same. If there are two numbers, the 1st is for the input(s), the second is for the output. If there are 3 numbers, the 1st is for the 1st input, the 2nd for the 2nd input, and the third is for the output.
* 24bit input must be sign-extended to 32bits; 24bit output will be sign-extended to 32bits.
* Some intrinsics can generate 32bit output, but very few can consume it; 32bit output will need to be consumed by the CPU directly.
* Intrinsics reserve hardware access via a mutex managed by the driver. Parameter ``keep_reservation`` can be set to true to avoid freeing and reserving the hardware between consecutive intrinsic invocations. It must be set to false on the last intrinsic.
* Some intrinsics are "compound intrinsics"; ie, they execute multiple commands to hardware. Rounding may be applied at any or all stages. Refer to the specific intrinsic's documentation for clarity.

.. _intrinsic_api_list:

Intrinsic Listing
*****************

The following intrinsics are currently supported. This list will grow in future releases. This table summarizes the available intrinsics; consult `/include/drivers/axon/axon_dsp_intrinsics.h` for parameter details.

Fast Fourier Transform (FFT) Intrinsics
=======================================

Axon NPU FFT requires 24bit complex input, with real/imaginary coefficients interleaved, and produces 24bit complex output of the same format.
Helper intrinsics can apply a window function to 16bit real input into 24bit complex input.

FFT intrinsics have a ``half_output`` parameter. When ``true``, only the 1st half of the output (below the Nyquist frequency) is written to the output buffer. 
This is purely for performance optimization.


.. list-table::
   :header-rows: 1

   * - Intrinsic
     - Description
     - Limitations
     - Axon software version
   * - nrf_axon_fft_24
     - Performs a 24bit FFT on 24bit input. Input and output are 24bit complex numbers.
     - Input and ouput lengths are a power of 2, and specifed as the log2 of the length. Minimum length is 64, maximum length is 512.
     - 1.0.0
   * - nrf_axon_fft_power_24
     - Compound intrinsic that performs a complex FFT, then squares and sums the ouput to produce real output.
     - Input and ouput lengths are a power of 2, and specifed as the log2 of the length. Minimum length is 64, maximum length is 512.
     - 1.0.0

Finite Impulse Response (FIR) Intrinsics
========================================

Some FIR intrinsics have filter coefficients in reverse order. This is due to the underlying hardware and how the FIR is implemented.
In the context of Axon FIR, "reversed" filter order means  

``x[n] = input[n-F]*filter[F-1] + input[n-F-1]*filter[F-2]+...input[n-1]*filter[0]``, where ``F`` is the length of the filter.

FIR intrinsics with "reversed" filters have the added property that the 1st valid output is offset ``F`` elements from the start of the output buffer.



.. list-table::
   :header-rows: 1

   * - Intrinsic
     - Description
     - Limitations
     - Axon software version
   * - nrf_axon_fir_24_24_24
     - | Performs an FIR with 24bit input, 24bit filters, and 24bit output.
       | Filter coefficients are reversed.
     - | length of the input (in elements) must be a multiple of 4, maximum of 512, and be at least 4 greater than filter_length
       | length of the filter (in elements) Must be a at least 12, a multiple of 4, and the last coefficient must be 0. 0 pad as necessary to meet these requirements.
     - 1.0.0
   * - nrf_axon_fir_24_16_24
     - | Performs an FIR with 24bit input, 16bit filters, and 24bit output.
       | Filter coefficients are reversed.
     - | length of the input (in elements) must be a multiple of 4, maximum of 512, and be at least 4 greater than filter_length
       | length of the filter (in elements) Must be a at least 12, a multiple of 4, and the last coefficient must be 0. 0 pad as necessary to meet these requirements.
     - 1.0.0
   * - nrf_axon_fir_2d_16_16_32_decimate
     - | Performs an FIR with 16bit input, 16bit filters, and 32 bit, decimated output.
       | Filter coefficients are not reversed.
       | Input dimensions are expressed in height x width, where width doubles as the output decimation factor.
     - | width of the input (in elements) must be greater than 1, and less than 17.
       | height of the input (and length of the output) must be less than 1024.
     - 1.0.0
   * - nrf_axon_fir_2d_16_16_24_decimate
     - | Performs an FIR with 16bit input, 16bit filters, and 24 bit, decimated output.
       | Filter coefficients are not reversed.
       | Input dimensions are expressed in height x width, where width doubles as the output decimation factor.
     - | width of the input (in elements) must be greater than 1, and less than 17.
       | height of the input (and length of the output) must be less than 1024.
     - 1.0.0
   * - nrf_axon_fir_16_16_32_decimate
     - | Performs an FIR with 16bit input, 16bit filters, and 32 bit, decimated output.
       | Filter coefficients are not reversed.
       | Input dimensions are expressed as length, with a separate decimation parameter.
       | It is preferred to use a 2D fir if possible. If the input length is not evenly divisible by the decimation factor, consider using a buffer sized to fit the 2D version and using the 2D version.
     - | Length of the input (in elements) must be greater than 4, and less than 513.
       | Length of the filter must be less than 33.
       | Decimation factor must be less than 17.
     - 1.0.0
   * - nrf_axon_fir_cplx_2d_16_16_24_decimate
     - | Performs an FIR with a complex filter with 16bit input, 16bit filters, and 24 bit, decimated output.
       | Filter coefficients are not reversed.
       | Filter coefficients are arranged with real coefficients first, followed by imaginary coefficients (ie, 2 back-to-back vectors)
       | Output values mirror filter coefficients; real coefficients followed by imaginary coefficients.
       | Input dimensions are expressed in height x width, where width doubles as the output decimation factor.
     - | width of the input (in elements) must be greater than 1, and less than 17.
       | height of the input (and length of the output) must be less than 1024.
     - 1.0.0
   * - nrf_axon_fir_cplx_2d_16_16_32_decimate
     - | Performs an FIR with a complex filter with 16bit input, 16bit filters, and 32 bit, decimated output.
       | Filter coefficients are not reversed.
       | Filter coefficients are arranged with real coefficients first, followed by imaginary coefficients (ie, 2 back-to-back vectors)
       | Output values mirror filter coefficients; real coefficients followed by imaginary coefficients.
       | Input dimensions are expressed in height x width, where width doubles as the output decimation factor.
     - | width of the input (in elements) must be greater than 1, and less than 17.
       | height of the input (and length of the output) must be less than 1024.
     - 1.0.0

Miscelaneous Intrinsics
=======================


.. list-table::
   :header-rows: 1

   * - Intrinsic
     - Description
     - Limitations
     - Axon software version
   * - nrf_axon_sqrt_24
     - Performs an element-wise, vector square root with 24bit input, and 24bit output.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_logn_11p12
     - | Performs an natural log with 24bit, q11.12 input and 24bit, q11.12 output.
       | If the input is not q11.12, remember your 8th grade algebra that log(x/y) = log(x) - log(y), so subtract the log of the radix difference from the result.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 2.
     - 1.0.0
   * - nrf_axon_exp_11p12
     - | Performs an ``e`` raised to the power of ``x`` 24bit, q11.12 input and 24bit, q11.12 output.
       | If the input is not q11.12, remember your 8th grade algebra that x^(y+z) = x^y * x^z, so multiply the result by e raised to the radix difference.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 2.
     - 1.0.0
   * - nrf_axon_saturate_32_24
     - Saturates a 32 bit input vector to 24 bit.
     - length of the input (in elements) has a maximum of 1024.
     - 1.0.0
   * - nrf_axon_saturate_32_24
     - Saturates a 32 bit input vector to 8 bit.
     - length of the input (in elements) has a maximum of 1024.
     - 1.0.0

Linear Algebra Intrinsics
=========================

.. list-table::
   :header-rows: 1

   * - Intrinsic
     - Description
     - Limitations
     - Axon software version
   * - nrf_axon_xspys_24_24_24
     - | "X squared plus Y squared" 
       | Each element of vectors X and Y are squared, then added to each other.
       | Input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_abs_24_24_24
     - | Returns the absolute of two vectors.
       | Each element of vectors X and Y is squared, summed, then has its square root taken.
       | Input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_xspys_24_24_24_input_stride2
     - | "X squared plus Y squared, skip every other element on the inputs"
       | Each element of vectors X and Y is squared, then added to each other.
       | By skipping every other element, a complex vector with real and imaginary components interleaved can be summed. The 1st input is the beginning of the vector, the 2nd input is offset by 1 element.
       | Input and output vectors are 24 bit.
     - | Length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
       | Length is not affected by the input stride = 2. 
     - 1.0.0
   * - nrf_axon_xsmys_24_24_24
     - | "X squared minus Y squared" 
       | Each element of vectors X and Y is squared, then Y-squared is subtracted from X-squared.
       | Input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_xty_24_24_24
     - | "X times Y" 
       | Each element of vector X is multiplied by the corresponding element of vector Y.
       | Input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_xty_16_16_32
     - | "X times Y" 
       | Each element of vector X is multiplied by the corresponding element of vector Y.
       | Input vectors are 16 bit, output vector is 32 bit.
     - length of the input (in elements) has a maximum of 512.
     - 1.0.0
   * - nrf_axon_xty_16_16_32_output_stride
     - | "X times Y, each output skips over output_stride elements" 
       | Each element of vector X is multiplied by the corresponding element of vector Y.
       | Input vectors are 16 bit, output vector is 32 bit.
       | This function is useful for FFT pipeline. Set output_extra_stride to 1. Input x is real, 16bit input input y is a a window function. The output will be spaced out to match the FIR input requirements. (Imaginary coefficients can be zeroed out using nrf_axon_memset_32_output_stride).
     - length of the input (in elements) has a maximum of 512.
     - 1.0.0
   * - nrf_axon_xpy_24_24_24
     - | "X plus Y" 
       | Each element of vector X is summed with the corresponding element of vector Y.
       | Input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_xmy_24_24_24
     - | "X minus Y" 
       | Each element of vector Y is subtracted from the corresponding element of vector X.
       | Input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_axpb_24_24
     - | "a * X + b" 
       | Each element of vector X is multipled by scalar a, then summed with scalar b.
       | Scalars a and b are passed by reference.
       | Input scalars and input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_axpb_2d_8_16
     - | "2 dimensional a * X + b" 
       | Input dimensions are expressed as height x width. This enables better performance from Axon.
       | Each element of vector X is multipled by scalar a, then summed with scalar b.
       | Scalars a and b are passed by value.
       | Input scalars are 32bit, input vector is 8bit, and output vector is 16 bit.
     - | height of the input (in rows) should be greater than 2, and less than input width.
       | width of input (in elements) should be a multiple of 2. If it is not, each row of output will have 16bits of padding.
     - 1.0.0
   * - nrf_axon_axpb_2d_16_16
     - | "2 dimensional a * X + b" 
       | Input dimensions are expressed as height x width. This enables better performance from Axon.
       | Each element of vector X is multipled by scalar a, then summed with scalar b.
       | Scalars a and b are passed by value.
       | Input scalars are 32bit, input vector is 16 bit, and output vector is 16 bit.
     - | height of the input (in rows) should be greater than 2, and less than input width.
       | width of input (in elements) should be a multiple of 2. If it is not, each row of output will have 16bits of padding.
     - 1.0.0
   * - nrf_axon_axpby_24_24_24
     - | "a * X plus b * Y" 
       | Each element of vector X is multipled by the scalar a, and summed with the corresponding product of vector Y and scalar b.
       | Scalars a and b are passed by reference.
       | Input scalars and input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0
   * - nrf_axon_xs_24_24
     - | "x squared" 
       | Each element of vector X is multipled by itself.
       | Input and output vectors are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 4.
     - 1.0.0

Accumlating Intrinsics
======================

.. list-table::
   :header-rows: 1

   * - Intrinsic
     - Description
     - Limitations
     - Axon software version
   * - nrf_axon_mar_24_24_32
     - | "Multiply, accumulate, recurrsive" (i.e, dot product)
       | Each element of vectors X and Y are multiplied together, then added to an accumulating sum.
       | Input vectors are 24 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_mar_16_24_32
     - | "Multiply, accumulate, recurrsive" (i.e, dot product)
       | Each element of vectors X and Y are multiplied together, then added to an accumulating sum.
       | 1st input vector is 16 bit, 2nd input vector is 24 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_mar_16_16_32
     - | "Multiply, accumulate, recurrsive" (i.e, dot product)
       | Each element of vectors X and Y are multiplied together, then added to an accumulating sum.
       | Input vectors are 16 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_mar_24_24_24
     - | "Multiply, accumulate, recurrsive" (i.e, dot product)
       | Each element of vectors X and Y are multiplied together, then added to an accumulating sum.
       | Input vectors and output scalar are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_mar_16_24_24
     - | "Multiply, accumulate, recurrsive" (i.e, dot product)
       | Each element of vectors X and Y are multiplied together, then added to an accumulating sum.
       | 1st input vector is 16 bit, 2nd input vector and output scalar are 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_mar_16_16_24
     - | "Multiply, accumulate, recurrsive" (i.e, dot product)
       | Each element of vectors X and Y are multiplied together, then added to an accumulating sum.
       | Input vectors are 16 bit, output scalar is 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_marx_24_32
     - | "Multiply, accumulate, recurrsive, only load the x input" (i.e, dot product)
       | Re-uses the 2nd input from the previous nrf_axon_mar_* operation ('*' indicates a wild-card; any of the functions that begin with nrfx_axon_mar_).
       | Caller must guarantee that the previous operation was one of the nrf_axon_mar_* or nrf_axon_mar_* (by having invoked this function directly with keep_reservation=true)
       | Each element of vector X is multplied with the corresponding vector Y loaded by the most recent call to nrf_axon_mar_*, then added to an accumulating sum.
       | Input vector is 24 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_marx_16_32
     - | "Multiply, accumulate, recurrsive, only load the x input" (i.e, dot product)
       | Re-uses the 2nd input from the previous nrf_axon_mar_* operation ('*' indicates a wild-card; any of the functions that begin with nrfx_axon_mar_).
       | Caller must guarantee that the previous operation was one of the nrf_axon_mar_* or nrf_axon_mar_* (by having invoked this function directly with keep_reservation=true)
       | Each element of vector X is multplied with the corresponding vector Y loaded by the most recent call to nrf_axon_mar_*, then added to an accumulating sum.
       | Input vector is 16 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_marx_24_24
     - | "Multiply, accumulate, recurrsive, only load the x input" (i.e, dot product)
       | Re-uses the 2nd input from the previous nrf_axon_mar_* operation ('*' indicates a wild-card; any of the functions that begin with nrfx_axon_mar_).
       | Caller must guarantee that the previous operation was one of the nrf_axon_mar_* or nrf_axon_mar_* (by having invoked this function directly with keep_reservation=true)
       | Each element of vector X is multplied with the corresponding vector Y loaded by the most recent call to nrf_axon_mar_*, then added to an accumulating sum.
       | Input vector and output scalar are 32 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_marx_16_24
     - | "Multiply, accumulate, recurrsive, only load the x input" (i.e, dot product)
       | Re-uses the 2nd input from the previous nrf_axon_mar_* operation ('*' indicates a wild-card; any of the functions that begin with nrfx_axon_mar_).
       | Caller must guarantee that the previous operation was one of the nrf_axon_mar_* or nrf_axon_mar_* (by having invoked this function directly with keep_reservation=true)
       | Each element of vector X is multplied with the corresponding vector Y loaded by the most recent call to nrf_axon_mar_*, then added to an accumulating sum.
       | Input vector is 16 bit, output scalar is 24 bit.
     - length of the input (in elements) must be a multiple of 2, maximum of 512, and be at least 8.
     - 1.0.0
   * - nrf_axon_acc_24_32
     - | "accumulate"
       | Each element of vector X is added to an accumulating sum.
       | Input vector is 24 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0
   * - nrf_axon_acc_16_32
     - | "accumulate"
       | Each element of vector X is added to an accumulating sum.
       | Input vector is 16 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0
   * - nrf_axon_acc_24_24
     - | "accumulate"
       | Each element of vector X is added to an accumulating sum.
       | Input vector and output scalar are 24 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0
   * - nrf_axon_acc_16_24
     - | "accumulate"
       | Each element of vector X is added to an accumulating sum.
       | Input vector is 16 bit, output scalar is 24 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0
   * - nrf_axon_l2norm_24_32
     - | "l2 norm"
       | Each element of vector X is squared then added to an accumulating sum.
       | Input vector is 24 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0
   * - nrf_axon_l2norm_16_32
     - | "l2 norm"
       | Each element of vector X is squared then added to an accumulating sum.
       | Input vector is 16 bit, output scalar is 32 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0
   * - nrf_axon_l2norm_24_24
     - | "l2 norm"
       | Each element of vector X is squared then added to an accumulating sum.
       | Input vector and output scalar are 24 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0
   * - nrf_axon_l2norm_16_24
     - | "l2 norm"
       | Each element of vector X is squared then added to an accumulating sum.
       | Input vector is 16 bit, output scalar is 24 bit.
     - length of the input (in elements) must be a multiple of 4, maximum of 1024, and be at least 8.
     - 1.0.0

DMA Intrinsics
==============

.. list-table::
   :header-rows: 1

   * - Intrinsic
     - Description
     - Limitations
     - Axon software version
   * - nrf_axon_memset_32_output_stride
     - | "32bit memset" 
       | Each element of output is set to the passed value.vector X is multiplied by the corresponding element of vector Y.
       | Input scalar is 16 bit, output vector is 32 bit.
       | This function is useful for FFT pipeline to zero-out the imaginary input coefficients, skipping over the real input coefficients. Set output_extra_stride to 1 and set value to 0.
     - length of the input (in elements) has a maximum of 512.
     - 1.0.0
   * - nrf_axon_dma_2d
     - | "2 Dimensional Direct Memory Access" 
       | Moves input data to the output location.
       | Input dimensions are expressed as height x width. This enables better performance from Axon.
       | input and output vectors are 8bit.
     - | height of the input (in rows) should be greater than 2, and less than input width.
       | width of input (in elements) must be less than 1025, and should be a multiple of 4. If it is not, each row of output will be padded to the next 32bit boundary.
     - 1.0.0
     

