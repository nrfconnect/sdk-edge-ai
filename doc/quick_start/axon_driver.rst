.. _quick_start_axon_driver:
.. _setup_axon:

Axon driver
###########

.. contents::
   :local:
   :depth: 2

The following guide outlines the requirements that you need to fulfill before you start working with the Axon NPU using Axon API directly, without the abstraction layer of |EAILib| API.
It also explains how to use the Axon NPU driver API to run TensorFlow Lite models directly on the Axon NPU.
This solution is ideal if you require low‑level control over inference execution, memory usage, and system integration.

To follow this guide, you should be familiar with embedded systems development and C‑based APIs.
Compared to higher‑level frameworks such as |EAILib|, using the driver API requires more manual setup but enables finer control over performance and resource utilization.

After completing this guide, you will have compiled a TensorFlow Lite model for the Axon NPU and deployed a custom application that performs inference using the Axon driver API.

.. _axon_requirements_hardware:

Hardware requirements
*********************

Axon NPU library is included as part of the :ref:`lib_axon` and provided as compiled binaries for Cortex-M33F architectures.
Axon NPU is currently available on the `nRF54LM20B`_ device.

.. _axon_requirements_software:

Software requirements
*********************

To start working with the Axon NPU, complete the setup based on your use case:

* If you want to deploy models on the device, you just need to complete :ref:`setup_sdk` to install |NCS| and toolchain.
* If you want to prepare models for deployment, you only need to set up a Python environment to run the :ref:`axon_npu_tflite_compiler`.
  Follow instructions in :ref:`axon_setup_compiler` to set up the environment.

.. _axon_setup_compiler:

Setting up Axon TFlite Compiler
===============================

Before you can run the :ref:`axon_npu_tflite_compiler`, you need to set up a Python environment with the required dependencies.
The executor of the compiler is compatible with Python ``3.11``.

You can set up the Python environment using one of the methods below.

.. tabs::

   .. group-tab:: Python virtual environment

      Using a virtual environment is strongly recommended to isolate dependencies.
      You can use any virtual environment tool you prefer.
      This section shows one example using Miniforge (Conda):

      1. Install `Miniforge`_

      #. Ensure the Conda :file:`scripts` directory is added to your system ``PATH``, for example, :file:`C:/Users/<user>/AppData/Local/miniforge3/Scripts`.

      #. Create a new environment with the supported Python version:

         .. code-block:: console

            conda create -n <env_name> python=3.11

      #. Activate the environment.
         All installation and execution commands must be run from the activated environment:

         .. code-block:: console

            conda activate <env_name>

      #. Install the required Python packages using the :file:`requirements.txt` file:

         .. code-block:: console

            cd tools/axon/compiler/scripts
            pip install -r requirements.txt

      .. note::

         On macOS, you may encounter the ``ERROR: No matching distribution found for tensorflow==2.15.1`` error.
         To fix it, install TensorFlow using Conda instead:

         .. code-block:: console

            conda install -c conda-forge tensorflow=2.15.1

      Once you complete the setup, you can try running the compiler by following instructions in :ref:`axon_npu_tflite_compiler`.

   .. group-tab:: Docker

      Docker provides a fully isolated way to run the compiler without installing dependencies locally.

      Before using Docker with the compiler, you must install it on the system.
      You should also verify that Docker is working correctly by building and running a simple Docker container.

      The following links provide introductory material and best-practice guidance for Docker:

      * `A beginner's guide to Docker`_
      * `Creating the Dockerfile`_
      * `Intro Guide to Dockerfile Best Practices`_

      Once you have installed and verified Docker, you can follow :ref:`axon_npu_tflite_compiler_docker` to build and run a Docker container for the Python compiler executor.

   .. group-tab:: Podman

      Podman is a daemonless alternative to Docker.
      Follow the steps below to set up Podman and run the compiler in a Podman container:

      1. Install Podman by following the `Podman installation guide`_.
      #. Set up and run a `simple container with Podman <Setting up Podman container_>`_

      Once you have installed and verified Podman, you can follow :ref:`axon_npu_tflite_compiler_podman` to build and run a Podman container for the Python compiler executor.

.. _quick_start_axon_driver_model_compilation:

Model compilation
*****************

With the compiler environment ready, you can transform your TensorFlow Lite model into Axon-optimized code and verify that inference produces correct results.

.. rst-class:: numbered-step

Compile your model
==================

The Axon compiler analyzes your model's operations, maps them to hardware accelerators, and generates efficient code for the NPU.
Whether you're using a pre-trained model or one you have trained yourself, you will need to run it through this compilation process.

Follow the :ref:`axon_npu_tflite_compiler_setup_executor` instructions to transform your TFLite model into an Axon-optimized model.

.. rst-class:: numbered-step

Verify compilation
==================

Test your compiled model to ensure it works correctly before integrating it into your application.

Run :ref:`test_nn_inference` to confirm your compiled model produces correct results.
This validation step checks for compilation issues early in the development process.

.. _quick_start_axon_driver_app_development:

Application development
***********************

Once you have a compiled model, you must integrate it into your embedded application.
Use the Axon driver API to load your model, manage memory, and execute inference directly on the NPU hardware.

.. rst-class:: numbered-step

Get compatible hardware
=======================

The Axon driver requires direct access to NPU hardware.
Obtain a development board with `Axon NPU`_.
Keep in mind the NPU is only available on select Nordic devices, so verify compatibility before starting.

.. rst-class:: numbered-step

Set up the Axon driver
=======================

Install the Axon runtime library and driver components on your development system.
Follow the :ref:`Axon driver setup <setup_axon>` instructions to prepare your environment for building and deploying Axon applications.

.. rst-class:: numbered-step

Verify your setup
=================

Before developing your own application, verify that everything is working correctly.
Run the :ref:`sample_hello_axon` sample application to confirm the driver can communicate with the NPU hardware.

.. note::
   Successfully running this sample means your development environment is ready.
   Any issues at this stage are typically related to hardware setup or driver installation.

.. rst-class:: numbered-step

Develop your application
========================

With your environment set up and model compiled, you can start building your Axon application.
Use the Axon driver API to load your compiled model, manage memory buffers, and execute inference.

.. tip::
   Start by modifying the :ref:`sample_hello_axon` sample to understand the basic API flow before building your custom application from scratch.

Follow the :ref:`ug_axon_integration` guide for detailed instructions on:

* Initializing the Axon driver
* Initializing your compiled model for synchronous or asynchronous inference
* Executing inference and handling results
* Integrating the model into your application

.. rst-class:: numbered-step

Deploy and optimize
===================

Build your application and flash it to your Nordic device.

.. include:: /includes/build_and_run_general.txt

Monitor performance metrics like inference time and current consumption to ensure your application meets requirements.
The direct driver access gives you the control needed to fine-tune performance for demanding embedded AI applications.

Next steps
**********

See further documentation:

* Add runtime monitoring of model outputs with :ref:`nrf_edgeai_obsv_lib`.
