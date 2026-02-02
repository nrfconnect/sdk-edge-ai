
.. _lib_axon:

Axon NPU: Hardware Accelerated Machine Learning
###############################################

.. contents::
   :local:
   :depth: 2

Overview
********

The Axon NPU accelerates neural networks.
This repository contains the driver, sample, simulator, and tool chain files to support Axon NPU.
Applications can be created to target the SoC (running zephyr) or a software simulator running on the host machine.

Neural Net Workflow
*******************

The neural net compiler for Axon NPU is provided in :file:`compiler/scripts`.
It has two components; an executor script and a compiler shared object/dll.
The compiler can work with a .tflite file (bare minimum) or the complete Keras model. The complete model with a test data set is needed if the user wants to measure quantization loss.
The user is responsible for populating a configuration .yml file with various parameters. 
The compiler will produce 3 header files: 

* :file:`nrf_axon_model_<model_name>_.h` compiled model header file.
* :file:`nrf_axon_model_<model_name>_test_vectors_.h` optional compiled-in test vectors.
* :file:`nrf_axon_model_<model_name>_layers_.h` optional header file with each layer as a separate model. 
  (<model_name> is the short named assigned to the model in the configuration .yaml file.)

The compiler will also report the model's memory requirements and provide a performance estimate (if test data was provided).

The quickest way to verify the model runs properly and to get precise performance numbers is to use the test application :file:`samples/test_nn_inference`.

See :file:`compiler/scripts/README.rst` for more details.

Building Applications
*********************

All applications under :file:`samples` can target zephyr or the simulator. 
The simulator is simply a static library that is compiled into a console application. It is not cycle exact with hardware, but is bit exact and provides performance estimates.

Zephyr Applications
-------------------

Kconfig dependencies
====================

.. code-block:: console

   CONFIG_NRF_AXON=y
   CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE=140000 # set this to the maximum value of all Axon models included in the build.

To build the zephyr application, execute "west build..." in the application's root.

Simulator Applications
----------------------

To build a simulator application, use the CMakeLists.txt file under <application>/simulator. The simulator application links with various pre-compiled libraries, so it is required to install the same toolchain the pre-compiled libraries were compiled with.


* For Windows, install and use the MSVC tool chain with CMake extensions in VS Code.
* For Linux, install and use the GCC tool chain.
  
  * Create a folder under simulator build_cli_linux and move to this directory.
  * Generate the Cmake files: cmake -DCMAKE_C_COMPILER:FILEPATH=gcc -DCMAKE_CXX_COMPILER:FILEPATH=g++ -DCMAKE_C_FLAGS:STRING=-fPIC -S.. -B. -G Ninja
  * Build the image: 

::

   cmake --build .

* For MacOS, install and use the AppleClang tool chain.
  
  * Create a folder under simulator build_cli_macos and move to this directory.
  * Generate the Cmake files: cmake -DCMAKE_C_COMPILER:FILEPATH=clang -DCMAKE_CXX_COMPILER:FILEPATH=clang++ -DCMAKE_C_FLAGS:STRING=-fPIC -S.. -B. -G Ninja
  * Build the image: 

:: 

   cmake --build .

Each application's CMakeList.txt file adds the repos root directory. 
At each directory level, CMakeList.txt files add local subdirectories, conditioned on kconfig variables.

For zephyr apps, the kconfig values are declared in the .prj file. For simulator apps, the kconfig values are declared in the <application>/simulator/CMakeLists.txt file.
