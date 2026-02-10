.. _test_nn_inference:

Test: NN Inference
##################

.. contents::
   :local:
   :depth: 2

The Test NN Inference application provides a simple way to run and validate a compiled neural network model on an Axon‑enabled target.

Requirements
************

TBA

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

   * Set ``CONFIG_AXON_MODEL_NAME`` to the model name (for example, ``tinyml_kws``).
     This value is used to include the correct header file and resolve model symbols.
   * Set ``CONFIG_AXON_INTERLAYER_BUFFER_SIZE`` to a value large enough for the model.
     Use the value defined by ``AXON_MODEL_<model_name>_MAX_IO_BUFFER_USED`` in the :file:`axon_model_<model_name>.h` file.
     A value of around ``115000`` is typically sufficient.

#. Edit the following macros in :file:`src/nrf_axon_app_nn_test_nn_inference.c` to control how inference is performed:
  
   * ``INCLUDE_VECTORS`` - When set to ``0``, test vectors are excluded and no inference is performed.
     This mode is useful for measuring the image size without test data.
   * ``AXON_MINIMUM_TEST_VECTORS`` - When set to ``1``, only a single end‑to‑end test vector is included.
     This produces the smallest application that still performs a valid inference.
   * ``AXON_LAYER_TEST_VECTORS`` - When set to ``1``, individual layer test vectors are included and executed.
   * ``AXON_LAYER_TEST_START_LAYER`` and ``AXON_LAYER_TEST_END_LAYER`` - Use these to limit testing to a specific range of layers, which can help reduce image size or focus debugging on specific layers.

#. Build the application by running the appropriate commands for your chosen build method:
  
   * For a command‑line Zephyr build, run the ``west build`` from the application directory.
     Flash the application to the device and monitor the UART output for test results.
   * For a simulator build in Visual Studio Code, use the CMake extension to build and run the application directly.

     Sample output should be as follows:

    .. code-block:: console

        TEST:   test_nn_inference_tinyml_kws       START CASE NO   0
        Test inference tinyml_kws vector 0 layers 0-11
        output bit exact!
        model tinyml_kws inference: ndx 7, label STOP, score 266992197, profiling ticks 0

        TEST:   test_nn_inference_tinyml_kws       CASE NO 0       RESULT: PASS
        
        TEST:   test_nn_inference_tinyml_kws       COMPLETE        PASS COUNT      1       FAIL COUNT      0
        Exit Simulator!

Dependencies
************

This test uses the following Edge AI Add-on libraries:

* :ref:`Axon driver <axon_driver>`
