.. _test_axon_intrinsics:

Test: DSP intrinsics
####################

.. contents::
   :local:
   :depth: 2

This test demonstrates how to run and validate Axon NPU DSP intrinsic functions on an Axon-enabled target.

Requirements
************

The test supports the following development kits:

.. table-from-sample-yaml::

Overview
********

Each Axon DSP intrinsic function is called and its output is compared to the expected result.
Inputs and reference outputs are in the :file:`src/nrf_axon_app_test_dsp_intrinsics_vectors.h` file, and the API is documented in :c:group:`nrf_axon_dsp_intrinsics`.

Each intrinsic runs through the Axon driver in synchronous blocking mode.

It can be built either for Zephyr-based targets or for the Axon simulator.
For more information about the DSP intrinsics, see :ref:`axon_driver`.

Tested intrinsic groups
=======================

The test exercises the following intrinsic groups:

* Vector operations - ``xty``, ``xpy``, ``xmy``, ``xspys``, ``xsmys``, ``axpby``, ``axpb``, ``xs``, ``abs``, and ``acc``
* Memory and saturation - ``memset`` and ``saturate``
* FFT - ``fft_24`` and ``fft_power_24``
* FIR filtering - ``fir_24_24_24``, ``fir_24_16_24``, ``fir_2d_16_16_32_decimate``, and ``fir_cplx_2d_16_16_24_decimate``
* Math functions - ``sqrt``, ``logn``, ``exp``, and ``mar``
* 2D operations - ``dma_2d``, ``axpb_2d_8_16``, and ``axpb_2d_16_16``

Building and running
********************

This section describes how to configure, build, and run the test.
Select how you want to build the application:

.. tabs::

   .. group-tab:: Zephyr

      #. Ensure that the ``CONFIG_NRF_AXON`` Kconfig option is enabled in the :file:`prj.conf` file.
         The default configuration already enables the Axon driver.
      #. Run ``west build`` from the application directory.
      #. Flash the application to the device and monitor the UART output for test results.

   .. group-tab:: Simulator

      #. Install a CMake extension in Visual Studio Code and add the :file:`simulator` folder to your workspace.
      #. Build and run the application from :file:`simulator/CMakeLists.txt`.

Example output
**************

Both build targets share the same test logic and produce the same ``TEST: AXON_INTRINSICS`` output.
On Zephyr, you also see boot banners and a board name line before the test starts.

.. code-block:: console

    nrf_axon_app_test_dsp_intrinsics on nrf54lm20dk
    Start axon_app_intrinsics!
    TEST:   AXON_INTRINSICS CASE COUNT      56
    xty_16_16_32_output_stride_tests: 5 cases
    TEST:   AXON_INTRINSICS START CASE NO   0
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 0       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   1
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 1       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   2
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 2       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   3
    TEST:   AXON_INTRINSICS CASE NO 3       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   4
    TEST:   AXON_INTRINSICS CASE NO 4       RESULT: PASS
    memset_32_output_stride tests: 3 cases
    TEST:   AXON_INTRINSICS START CASE NO   5
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 5       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   6
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 6       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   7
    TEST:   AXON_INTRINSICS CASE NO 7       RESULT: PASS
    saturate_32_24_tests: 3 cases
    TEST:   AXON_INTRINSICS START CASE NO   8
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 8       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   9
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 9       RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   10
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 10      RESULT: PASS
    xty_16_16_32_tests: 4 cases
    TEST:   AXON_INTRINSICS START CASE NO   11
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 11      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   12
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 12      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   13
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 13      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   14
    TEST:   AXON_INTRINSICS CASE NO 14      RESULT: PASS
    fft_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   15
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 15      RESULT: PASS
    xspys_24_24_24_input_stride2_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   16
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 16      RESULT: PASS
    xspys_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   17
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 17      RESULT: PASS
    xsmys_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   18
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 18      RESULT: PASS
    mar_16_24_32_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   19
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 19      RESULT: PASS
    mar_16_24_24_tests: 3 cases
    TEST:   AXON_INTRINSICS START CASE NO   20
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 20      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   21
    marx_16_24
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 21      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   22
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 22      RESULT: PASS
    sqrt_24_tests: 4 cases
    TEST:   AXON_INTRINSICS START CASE NO   23
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 23      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   24
    TEST:   AXON_INTRINSICS CASE NO 24      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   25
    TEST:   AXON_INTRINSICS CASE NO 25      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   26
    TEST:   AXON_INTRINSICS CASE NO 26      RESULT: PASS
    logn_11p12_tests: 4 cases
    TEST:   AXON_INTRINSICS START CASE NO   27
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 27      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   28
    TEST:   AXON_INTRINSICS CASE NO 28      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   29
    TEST:   AXON_INTRINSICS CASE NO 29      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   30
    TEST:   AXON_INTRINSICS CASE NO 30      RESULT: PASS
    exp_11p12_tests: 4 cases
    TEST:   AXON_INTRINSICS START CASE NO   31
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 31      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   32
    TEST:   AXON_INTRINSICS CASE NO 32      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   33
    TEST:   AXON_INTRINSICS CASE NO 33      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   34
    TEST:   AXON_INTRINSICS CASE NO 34      RESULT: PASS
    xs_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   35
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 35      RESULT: PASS
    axpby_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   36
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 36      RESULT: PASS
    xty_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   37
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 37      RESULT: PASS
    xpy_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   38
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 38      RESULT: PASS
    xmy_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   39
    Verify ...  PASSED!
    TEST:   AXONRESULT:     PASS
    acc_16_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   40
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 40      RESULT: PASS
    fir_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   41
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 41      RESULT: PASS
    fir_24_16_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   42
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 42      RESULT: PASS
    axpb_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   43
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 43      RESULT: PASS
    fft_power_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   44
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 44      RESULT: PASS
    fir_2d_16_16_32_decimate_tests: 4 cases
    TEST:   AXON_INTRINSICS START CASE NO   45
    2d fir: 1024 input, 12 filter, decimation 2, profiling ticks 110
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 45      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   46
    2d fir: 1024input, 256 filter, decimation 4,    profiling ticks 122
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 46      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   47
    1d fir: 1024input, 256 filter, decimation 1,    profiling ticks 4217
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 47      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   48
    2d fir 16_16_24: 768 input, 256 filter, decimation 16,profiling ticks 60
    ,Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 48      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   49
    2d complex fir 16_16_24: 768 input, 256 filter, decimation 16,profiling ticks 69
    ,Verify ...  PASSED!
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 49      RESULT: PASS
    dma_2d_tests: 3 cases
    TEST:   AXON_INTRINSICS START CASE NO   50
    Verify ... PASSED!
    TEST:   AXON_INTRINSICS CASE NO 50      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   51
    Failed as expected, width 47 is not a multiple of 4. code=-2
    TEST:   AXON_INTRINSICS CASE NO 51      RESULT: PASS
    TEST:   AXON_INTRINSICS START CASE NO   52
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 52      RESULT: PASS
    abs_24_24_24_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   53
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 53      RESULT: PASS
    axpb_2d_8_16_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   54
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 54      RESULT: PASS
    axpb_2d_16_16_tests: 1 cases
    TEST:   AXON_INTRINSICS START CASE NO   55
    Verify ...  PASSED!
    TEST:   AXON_INTRINSICS CASE NO 55      RESULT: PASS
    TEST:   AXON_INTRINSICS COMPLETE        PASS COUNT      56      FAIL COUNT      0
    axon_app_intrinsics complete!

A successful run ends with ``COMPLETE PASS COUNT 56 FAIL COUNT 0``.

* ``Verify ... PASSED!`` means the intrinsic output matched the expected reference vector.
* ``Failed as expected`` indicates a negative test case where invalid parameters were rejected.
  These cases still report ``RESULT: PASS``.
* Some FIR test cases also print ``profiling ticks`` for performance reference.

Dependencies
************

This test uses the following Edge AI Add-on library:

* :ref:`Axon driver <axon_driver>`

The test calls the following function:

* :c:group:`nrf_axon_dsp_intrinsics`
