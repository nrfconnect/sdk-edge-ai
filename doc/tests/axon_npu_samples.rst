
.. _tests_axon_npu_samples:

Axon NPU Tests
##############

.. contents::
   :local:
   :depth: 2

This page contains test samples that demonstrate how to use the Axon NPU for acceleration tasks.

Overview
********

The samples cover neural network inference, DSP‑style algorithms for feature extraction, and combined use cases that integrate both approaches.

Each sample can be built either for a Zephyr‑based target or for the Axon simulator. 
The simulator is not a standalone tool or dedicated hardware. 
Instead, it is provided as a static library that is linked into a host‑based console application. 
While it is not cycle‑accurate compared to hardware, it is computationally bit‑exact and can be used to estimate performance.

Available tests are listed below:

.. toctree::
   :maxdepth: 1

   test_nn_inference.rst

Building applications
*********************

All applications under :file:`../samples` can target zephyr or the simulator:

* The simulator is simply a static library that is compiled into a console application. 
  It is not cycle exact with hardware, but is computationally bit-exact and provides performance estimates.
  For simulator apps, the kconfig values are declared in the :file:`<application>/simulator/CMakeLists.txt` file.
* To build a zephyr application, execute the ``west build`` command in the application's root.
  For zephyr applications, the Kconfig values are declared in the :file:`.prj` file. 
* To build a simulator application, use the :file:`CMakeLists.txt` file under :file:`<application>/simulator`. 
  If you are using CMake extensions in Visual Studio Code, add the simulator folder to the workspace.
  Each application's :file:`CMakeLists.txt` file adds the repos root directory.
  At each directory level, :file:`CMakeLists.txt` files add local subdirectories, conditioned on Kconfig variables.

Other applications
==================

This section lists the available `Axon NPU test applications <https://github.com/nrfconnect/sdk-axons-ml/tree/main/samples_>`_ and briefly describes what each one is used for.

axon_fe_mfcc
------------

This application demonstrates the use of the Axon MFCC (Mel‑frequency cepstral coefficients) implementation.
It is intended to show the general algorithm development flow on Axon, with MFCC as a concrete example.
Two simulator build targets are available:

* Compile
* Calculation

For Zephyr targets, only the calculation configuration is supported.

axon_intrinsics
---------------

This application exercises each available axon_dsp intrinsic and compares the results against known reference values.
It serves as a basic validation tool to verify that individual intrinsics behave as expected.
The set of supported intrinsics will expand over time, and new intrinsics will be added to this application as they become available.

include_models
--------------

This directory contains sample compiled model header files.
It is added to the include path of the test_nn_inference and kws_inference applications, which means that any model placed in this directory is automatically available to those applications.
