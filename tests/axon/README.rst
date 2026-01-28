
.. _nrf_axon_samples:

Axon NPU Samples
###############################################

.. contents::
   :local:
   :depth: 2

Overview
********

These samples demonstrate a variety of ways axon acceleration can be used. This includes neural network inference, dsp-like algorithms (used for feature extraction), and combining the two into a use-case.
Each sample can be built for both Zephyr and the axon simulator. (The simulator is not a dedicated piece of hardware or software, it is a static library that is included in the build application.)


Building Applications
*********************
All applications under `<../samples>`_ can target zephyr or the simulator. 
The simulator is simply a static library that is compiled into a console application. It is not cycle exact with hardware, but is computationally bit-exact and provides performance estimates.

To build a zephyr application, execute "west build..." in the application's root.

To build a simulator application, use the CMakeLists.txt file under <application>/simulator. If using CMake extensions in Visual Studio Code, add the simulator folder to the workspace. 

Each application's CMakeList.txt file adds the repos root directory. 
At each directory level, CMakeList.txt files add local subdirectories, conditioned on kconfig variables.

For zephyr apps, the kconfig values are declared in the .prj file. For simulator apps, the kconfig values are declared in the <application>/simulator/CMakeLists.txt file.

The applications' README.rst files provide more detail on each of the applications.

Subdirectories
**************

See the README.rst in each subdirectory for more details.

`<axon_fe_mfcc>`_ 
-------------------

Application that utilizes the axon MFCC implementation to demonstrate the algorithm development flow in general, and MFCC usage in particular.

There are 2 simulator build targets, compile and calculation. 

Only calculation can be built for Zephyr.

`<axon_intrinsics>`_
-------------------

Application that invokes each of the axon_dsp intrinsics and compares its results to known good values.
This is used by the development team as a basic hello word application verify intrinsics.
This number of supported intrinsics will grow over time, and will be added to this app.

`<test_nn_inference>`_
-------------------

Application that combines a compiled model with its test vectors with minimal. Useful for initial model validation and performance benchmarking on the target device.
kconfig is used to specify the model name, so it can be used to test any compiled model.

`<include_models>`_
-------------------

Sample compiled model header files are placed here. This is added to the path of test_nn_inference and kws_inference so any model placed in this folder will be available to those applications.




