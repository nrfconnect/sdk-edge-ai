.. _quick_start_axon_dsp_intrinsics:
.. _setup_axon_dsp_intrinsics:

Axon driver DSP intrinsics
##########################

.. contents::
   :local:
   :depth: 2

This guide explains how to run Axon NPU DSP intrinsic functions on an Axon-enabled target using the Axon driver API directly, without compiling a TensorFlow Lite model.

To follow this guide, you should be familiar with embedded systems development and C-based APIs.
After completing this guide, you will have built and run the :ref:`test_axon_intrinsics` application and understand how to integrate DSP intrinsics into your own application.

Hardware requirements
*********************

Axon NPU library is included as part of the :ref:`lib_axon` and provided as compiled binaries for Cortex-M33F architectures.
Axon NPU is currently available on the `nRF54LM20B`_ device.

Software requirements
*********************

Complete :ref:`setup_sdk` to install |NCS| and the development toolchain.
No Python compiler environment is required for DSP intrinsics development.

.. _quick_start_axon_dsp_intrinsics_app_development:

Application development
***********************

Use the Axon driver API to initialize the NPU and call DSP intrinsic functions directly on the hardware or in the simulator.

.. rst-class:: numbered-step

Get compatible hardware
=======================

The Axon driver requires direct access to NPU hardware.
Obtain a development board with `Axon NPU`_.

.. rst-class:: numbered-step

Set up the Axon driver
=======================

Install the Axon runtime library and driver components on your development system.
Complete :ref:`setup_sdk` to install |NCS| and the development toolchain.

.. rst-class:: numbered-step

Verify your setup
=================

Before developing your own application, verify that DSP intrinsic functions work correctly on your target.
Build and run :ref:`test_axon_intrinsics` and confirm that all test cases pass.

.. tabs::

   .. group-tab:: Zephyr

      #. Ensure that the ``CONFIG_NRF_AXON`` Kconfig option is enabled in the test :file:`prj.conf` file.
      #. Run ``west build`` from :file:`tests/axon/intrinsics`.
      #. Flash the application to the device and monitor the UART output for test results.

   .. group-tab:: Simulator

      #. Install a CMake extension in Visual Studio Code and add the :file:`tests/axon/intrinsics/simulator` folder to your workspace.
      #. Build and run the application from :file:`simulator/CMakeLists.txt`.

.. rst-class:: numbered-step

Develop your application
========================

Once the test passes, integrate DSP intrinsic functions into your application.
Follow the :ref:`ug_axon_dsp_intrinsics_integration` guide for driver initialization and calling intrinsic functions in your code.

.. note::

   See :ref:`supported_dsp_intrinsics` for the available intrinsic functions.
   If you also plan to combine intrinsics with compiled model inference, start from :ref:`sample_hello_axon` to understand the shared driver initialization flow.

.. rst-class:: numbered-step

Deploy and optimize
===================

Build your application and flash it to your Nordic device.

.. include:: /includes/build_and_run_general.txt

Next steps
**********

See further documentation:

* Follow :ref:`ug_axon_dsp_intrinsics_integration` for driver initialization and intrinsic function usage.
* See the :ref:`supported_dsp_intrinsics` for function parameters and usage notes.
