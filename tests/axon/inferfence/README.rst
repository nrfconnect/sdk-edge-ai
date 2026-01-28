.. _test_nn_inference:

Test NN Inference
#################

Overview
********

This application combines a model, test vectors, and optionally the individual layer models. It performs inference on the models and compares the
results with the expecited values. It is the quickest way to get a model running, verified, and profiled on a target device.
It can target zephyr or the axon simulator.

Only the Axon Simulator and Nordic devices with the Axon NPU can be targeted.

Building and Running
********************
#. To build the simulator application in VS Code, install CMake extension, and add the `<simulator>`_ folder to the workspace.
#. Select one of the sample models in `<../include_models>`_ or copy your own model header files to that folder. (See `compiler/scripts <../../compiler/scripts>`_ for details on compiling a model.)
#. Edit `<prj.conf>`_ (for zephyr) or `<simulator/CMakeLists.txt>`_ (for simulator) with the model parameters.
    #. Specify the model name in CONFIG_AXON_MODEL_NAME (ie, tinyml_kws). This is used to include the header file and specify the model symbol names.
    #. Set CONFIG_AXON_INTERLAYER_BUFFER_SIZE to a sufficiently large number specified by #define AXON_MODEL\_<model_name>_MAX_IO_BUFFER_USED in axon_model\_<model_name>_.h (115000 is typically enough.)
#. Select the level of execution by editing these macros in `<src/nrf_axon_app_nn_test_nn_inference.c>`_:
    #. `INCLUDE_VECTORS` : If set to 0, test vectors will not be included and no inferencing will occur. Building in this mode will show the size of the compiled image without the test vectors.
    #. `AXON_MINIMUM_TEST_VECTORS` : If set to 1, only 1 end-to-end vector will be compiled into the image. This will be the bare minimum application that can be built and still perform a test inference.
    #. `AXON_LAYER_TEST_VECTORS` : If set to 1, individual layer vectors will be included and inferenced. 
    #. `AXON_LAYER_TEST_START_LAYER`, `AXON_LAYER_TEST_START_LAYER` : Set these to specify which individual layers to inlclude (to limit image size and/or focus on specific layers).
#. For a command line zephyr build, 
    #. run west --build in this folder.
    #. Flash the build to the device.
    #. Monitor the UART for messages.
#. For a VS Code simulator build
    #. Use CMake extension to build and run the application. 


Sample Output (simulator build)
*******************************
.. code-block:: console

    TEST:   test_nn_inference_tinyml_kws       START CASE NO   0
    Test inference tinyml_kws vector 0 layers 0-11
    output bit exact!
    model tinyml_kws inference: ndx 7, label STOP, score 266992197, profiling ticks 0

    TEST:   test_nn_inference_tinyml_kws       CASE NO 0       RESULT: PASS
    
    TEST:   test_nn_inference_tinyml_kws       COMPLETE        PASS COUNT      1       FAIL COUNT      0
    Exit Simulator!

