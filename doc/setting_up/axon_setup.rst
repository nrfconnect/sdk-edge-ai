.. _setup_axon:

Setting up Axon
###############

.. contents::
   :local:
   :depth: 2

This page outlines the requirements that you need to fulfill before you start working with the Axon NPU using directly Axon API, without abstraction layer of |EAILib| API.

.. _axon_requirements_software:

Software requirements
*********************

To start working with the Axon NPU, you must:

1. Install the `nRF Connect SDK`_, including all its prerequisites and the |NCS| toolchain.
#. :ref:`Set up the Python environment to run the Axon TFlite compiler <axon_setup_compiler>`.
#. :ref:`Set up the environment to run the Axon Simulator <axon_setup_simulator>`.

.. _axon_setup_compiler:

Setting up Axon TFlite Compiler
===============================

Before you can run the Axon TFlite Compiler, you need to set up a Python environment with the required dependencies.
The executor of the compiler is compatible with Python ``v3.11``.

You can set up the Python environment using one of the methods below:

* :ref:`Setting up the Virtual environment <axon_setup_conda>`
* :ref:`Setting up Docker <axon_setup_docker>`
* :ref:`Setting up Podman <axon_setup_podman>`

Once you complete the setup, you can try running the compiler by following instructions in :ref:`lib_axon_zephyr`.

.. _axon_setup_conda:

Creating a virtual environment
----------------------------------

Using a virtual environment is strongly recommended to isolate dependencies.
You can use any virtual environment tool you prefer.
This section shows one example using Miniforge (Conda):

#. Install `Miniforge`_

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

.. _axon_setup_docker:

Setting up Docker
-----------------

Docker provides a fully isolated way to run the compiler without installing dependencies locally.

Before using Docker with the compiler, you must install it on the system.
You should also verify that Docker is working correctly by building and running a simple Docker container.

The following links provide introductory material and best-practice guidance for Docker:

* `A beginner's guide to Docker`_
* `Creating the Dockerfile`_
* `Intro Guide to Dockerfile Best Practices`_

Once you have installed and verified Docker, you can follow :ref:`axon_npu_tflite_compiler_docker` to build and run a Docker container for the Python compiler executor.

.. _axon_setup_podman:

Setting up Podman
-----------------

Podman is a daemonless alternative to Docker.
Follow the steps below to set up Podman and run the compiler in a Podman container:

1. Install Podman by following the `Podman installation guide`_.
#. Set up and run a `simple container with Podman <Setting up Podman container_>`_

Once you have installed and verified Podman, you can follow :ref:`axon_npu_tflite_compiler_podman` to build and run a Podman container for the Python compiler executor.

.. _axon_setup_simulator:

Setting up Axon Simulator environment
=====================================

The simulator application links with various pre-compiled libraries, so it is required to install the same toolchain the pre-compiled libraries were compiled with.

* For Windows, `Configure VS Code for Microsoft C++`_ to install MSVC toolchain.
  Additionally install CMake extension for VS Code.
* For Linux, install and use the GCC toolchain, CMake and Ninja.
* For MacOS, install and use the AppleClang toolchain, CMake and Ninja.

Once you complete the setup, you can try building a simulator application by following instructions in :ref:`lib_axon_simulator`.

.. _axon_requirements_hardware:

Hardware requirements
*********************

Axon NPU library is included as part of the |EAILib| package, provided as compiled binaries for ARM Cortex-M4F and Cortex-M33F architectures (see :ref:`nrf_edgeai_requirements_hardware`).
Axon NPU is available on selected Nordic Semiconductor's devices:

* `nRF54LM20B`_
