.. _test_nn_inference:

Test: NN Inference
##################

.. contents::
   :local:
   :depth: 2

The Test NN Inference application provides a simple way to run and validate a compiled neural network model on an Axon‑enabled target.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The application combines a model, test vectors, and optionally per‑layer test data, runs inference, and compares the results against expected values.
This application is intended as the fastest way to get a model running, verified, and profiled on a target device.
It can be built either for Zephyr‑based targets or for the Axon simulator.

.. note::
    This application only supports the Axon simulator and Nordic Semiconductor devices that include the Axon NPU.

Building and running
********************

This section describes how to configure, build, and run the application.

#. Select how you want to build the application:

   * To build the simulator application in Visual Studio Code, install a CMake extension and add the simulator folder to your workspace.
   * To build for Zephyr, use a standard Zephyr build workflow.

#. Select one of the sample models from :file:`/compiled_models` directory, or copy your own compiled model header files into that directory.
   For information about compiling models, see :ref:`axon_npu_tflite_compiler` documentation.

#. Configure the :file:`prj.conf` (for Zephyr builds) or :file:`simulator/CMakeLists.txt` (for simulator builds) with model parameters:

   * Set ``CONFIG_NRF_AXON_MODEL_NAME`` to the model name (for example, ``tinyml_kws``).
     This value is used to include the correct header file and resolve model symbols.
   * Set ``CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE`` to a value large enough for the model.
     Use the value defined by ``AXON_MODEL_<model_name>_MAX_IL_BUFFER_USED`` in the :file:`axon_model_<model_name>.h` file.
     A value of around ``115000`` is typically sufficient.

#. Edit the following macros in :file:`src/nrf_axon_app_nn_test_nn_inference.c` to control how inference is performed:

   * ``INCLUDE_VECTORS`` - When set to ``0``, test vectors are excluded and no inference is performed.
     This mode is useful for measuring the image size without test data.
   * ``AXON_MINIMUM_TEST_VECTORS`` - When set to ``1``, only a single end‑to‑end test vector is included.
     This produces the smallest application that still performs a valid inference.
   * ``AXON_LAYER_TEST_VECTORS`` - When set to ``1``, individual layer test vectors are included and executed.
   * ``AXON_LAYER_TEST_START_LAYER`` and ``AXON_LAYER_TEST_STOP_LAYER`` - Use these to limit testing to a specific range of layers, which can help reduce image size or focus debugging on specific layers.

#. Build the application by running the appropriate commands for your chosen build method:

   * For a command‑line Zephyr build, run the ``west build`` from the application directory.
     Flash the application to the device and monitor the UART output for test results.
   * For a simulator build in Visual Studio Code, use the CMake extension to build and run the application directly.

     Sample output should be as follows:

    .. code-block:: console

        *** Booting nRF Connect SDK v3.3.0-preview2-ede152ec210b ***
        *** Using Zephyr OS v4.3.99-4b6df5ff11b1 ***
        Hello world from nrf54lm20dk
        ticks per second 1000000
        Start Platform!
        Prepare and run Axon!

        TEST:   test_nn_inference_tinyml_kws    CASE COUNT      15

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   0
        Test inference tinyml_kws vector 0 FULL MODEL sync mode
        output bit exact!
        model tinyml_kws inference: ndx 7, label STOP, score 266802540, profiling ticks 5213

        TEST:   test_nn_inference_tinyml_kws    CASE NO 0       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   1
        Test inference tinyml_kws vector 1 FULL MODEL async mode
        output bit exact!
        model tinyml_kws inference: ndx 2, label LEFT, score 149726243, profiling ticks 5254

        TEST:   test_nn_inference_tinyml_kws    CASE NO 1       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   2
        Test inference tinyml_kws vector 2 FULL MODEL sync mode
        output bit exact!
        model tinyml_kws inference: ndx 6, label RIGHT, score 268434516, profiling ticks 5213

        TEST:   test_nn_inference_tinyml_kws    CASE NO 2       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   3

        Test inference tinyml_kws vector 0 layer 0
        output bit exact!
        profiling ticks 699

        TEST:   test_nn_inference_tinyml_kws    CASE NO 3       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   4

        Test inference tinyml_kws vector 0 layer 1
        output bit exact!
        profiling ticks 197

        TEST:   test_nn_inference_tinyml_kws    CASE NO 4       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   5

        Test inference tinyml_kws vector 0 layer 2
        output bit exact!
        profiling ticks 1028

        TEST:   test_nn_inference_tinyml_kws    CASE NO 5       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   6

        Test inference tinyml_kws vector 0 layer 3
        output bit exact!
        profiling ticks 197

        TEST:   test_nn_inference_tinyml_kws    CASE NO 6       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   7

        Test inference tinyml_kws vector 0 layer 4
        output bit exact!
        profiling ticks 1022

        TEST:   test_nn_inference_tinyml_kws    CASE NO 7       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   8

        Test inference tinyml_kws vector 0 layer 5
        output bit exact!
        profiling ticks 197

        TEST:   test_nn_inference_tinyml_kws    CASE NO 8       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   9

        Test inference tinyml_kws vector 0 layer 6
        output bit exact!
        profiling ticks 1023

        TEST:   test_nn_inference_tinyml_kws    CASE NO 9       RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   10

        Test inference tinyml_kws vector 0 layer 7
        output bit exact!
        profiling ticks 197

        TEST:   test_nn_inference_tinyml_kws    CASE NO 10      RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   11

        Test inference tinyml_kws vector 0 layer 8
        output bit exact!
        profiling ticks 1023

        TEST:   test_nn_inference_tinyml_kws    CASE NO 11      RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   12

        Test inference tinyml_kws vector 0 layer 9
        output bit exact!
        profiling ticks 150

        TEST:   test_nn_inference_tinyml_kws    CASE NO 12      RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   13

        Test inference tinyml_kws vector 0 layer 10
        output bit exact!
        profiling ticks 77

        TEST:   test_nn_inference_tinyml_kws    CASE NO 13      RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    START CASE NO   14

        Test inference tinyml_kws vector 0 layer 11
        output bit exact!
        profiling ticks 72

        TEST:   test_nn_inference_tinyml_kws    CASE NO 14      RESULT: PASS

        TEST:   test_nn_inference_tinyml_kws    COMPLETE        PASS COUNT      15      FAIL COUNT      0
        test_nn_inference complete!

Dependencies
************

This test uses the following Edge AI Add-on libraries:

* :ref:`Axon driver <axon_driver>`
