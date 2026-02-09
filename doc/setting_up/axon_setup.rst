.. _setup_axon:

Setting up Axon
###############

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with the Axon NPU using Axon API directly, without the abstraction layer of |EAILib| API.

.. _axon_requirements_hardware:

Hardware requirements
*********************

Axon NPU library is included as part of the :ref:`nrf_edgeai_lib` and provided as compiled binaries for Cortex-M33F architectures.
Axon NPU is currently available on the `nRF54LM20B`_ device.

.. _axon_requirements_software:

Software requirements
*********************

To start working with the Axon NPU, complete the setup based on your use case:

* If you want to deploy models on the device, you just need to complete :ref:`setup_sdk` to install |NCS|, toolchain, and |EAILib|.
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

            cd <project_root>/compiler/scripts
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
