.. _quick_start_axon_driver:

Axon driver
###########

.. contents::
   :local:
   :depth: 2

The following guide explains how to use the Axon NPU driver API to run TensorFlow Lite models directly on the Axon NPU.
It is ideal if you require low‑level control over inference execution, memory usage, and system integration.

To follow this guide, you should be familiar with embedded systems development and C‑based APIs.
Compared to higher‑level frameworks such as |EAILib|, using the driver API requires more manual setup but enables finer control over performance and resource utilization.

After completing this guide, you will have compiled a TensorFlow Lite model for the Axon NPU and deployed a custom application that performs inference using the Axon driver API.

Model compilation
*****************

Before you can deploy to the Axon NPU, you need a TensorFlow Lite model compiled specifically for Axon hardware.
The :ref:`axon_npu_tflite_compiler` transforms your TFLite model into an optimized format that leverages the NPU's specialized hardware.

.. rst-class:: numbered-step

Set up the compiler
===================

First, set up the Axon compiler's environment on your development system.
Follow the :ref:`Executor and compiler setup <axon_setup_compiler>` instructions to install the necessary tools.

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

Monitor performance metrics like inference time and power consumption to ensure your application meets requirements.
The direct driver access gives you the control needed to fine-tune performance for demanding embedded AI applications.
