.. _quick_start_axon_driver:

Axon Driver - Quick Start Guide
###############################

.. contents::
   :local:
   :depth: 2

Welcome to the Axon Driver quick start guide!
This guide is for developers who want maximum control and performance by working directly with the Axon NPU hardware.
Using the Axon driver API directly gives you low-level access to the NPU's capabilities, allowing for fine-tuned optimization and integration.

If you're comfortable with embedded systems programming and want to squeeze every bit of performance from the Axon NPU, this is the right path for you.
While this approach requires more technical expertise than using higher-level frameworks like |EAILib|, it offers the flexibility and control needed for advanced applications.

By the end of this guide, you'll have compiled a TensorFlow Lite model for Axon NPU and deployed a custom AI application using the driver API!

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
Whether you're using a pre-trained model or one you've trained yourself, you'll need to run it through this compilation process.

Follow the :ref:`axon_npu_tflite_compiler_setup_executor` instructions to transform your TFLite model into an Axon-optimized model.

.. rst-class:: numbered-step

Verify compilation
==================

Test your compiled model to ensure it works correctly before integrating it into your application.

Run :ref:`test_nn_inference` to confirm your compiled model produces correct results.
This validation step helps catch compilation issues early in the development process.

Application development
***********************

Now that you have a compiled model, it's time to integrate it into your embedded application.
You'll use the Axon driver API to load your model, manage memory, and execute inference directly on the NPU hardware.

.. rst-class:: numbered-step

Get compatible hardware
=======================

The Axon driver requires direct access to NPU hardware.
Obtain a development board with `Axon NPU`_ - the NPU is only available on select Nordic devices, so verify compatibility before starting.

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

.. tip::
   Successfully running this sample means your development environment is ready.
   Any issues at this stage are typically related to hardware setup or driver installation.

.. rst-class:: numbered-step

Develop your application
========================

With your environment set up and model compiled, you're ready to build your Axon-powered application!
The Axon driver API provides direct control over model loading, memory management, and inference execution.

Follow the :ref:`ug_axon_integration` guide for detailed instructions on:

* Initializing the Axon driver
* Initializing your compiled model for synchronous or asynchronous inference
* Executing inference and handling results
* Integrating the model into your application

.. tip::
   Start by modifying the :ref:`sample_hello_axon` sample to understand the basic API flow before building your custom application from scratch.
   This incremental approach helps you identify issues early.

.. rst-class:: numbered-step

Deploy and optimize
===================

Build your application, flash it to your Nordic device, and see your model running on the Axon NPU!

.. include:: /includes/build_and_run_general.txt

Monitor performance metrics like inference time and power consumption to ensure your application meets requirements.
The direct driver access gives you the control needed to fine-tune performance for demanding embedded AI applications.

Congratulations! You're now developing high-performance AI applications with direct Axon NPU access.
