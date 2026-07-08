.. _quick_start_edge_impulse:
.. _setup_edge_impulse:

Edge Impulse
############

.. contents::
   :local:
   :depth: 2

The following guide explains how to develop and deploy machine learning applications on Nordic Semiconductor devices using |EI|.
It is ideal if you want an end‑to‑end workflow for data collection, model training, and deployment on embedded targets.

To follow this guide, you should be familiar with basic embedded systems development.
The guide covers the steps required to collect data, train a model using |EI| tools, and deploy the resulting model to a Nordic device.

After completing this guide, you will have a machine learning application running on a Nordic Semiconductor device using |EI|.

.. _edge_impulse_requirements_hardware:

Hardware requirements
*********************

The support of specific hardware platforms depends on the sample application you wish to run.
For details, see the :ref:`edge_impulse_samples` page, or refer to individual samples' :file:`sample.yaml` files in the :file:`samples/edge_impulse/<sample>/` directory.

|EI| SDK is provided in form of source code and can be built during the application build process for a hardware architecture of the target device.
This means SDK and models can be built for any Nordic Semiconductor's device with ARM Cortex-M4F and Cortex-M33F architectures.

You can also deploy |EI| models utilizing Axon NPU, which is designed to accelerate machine learning inference on selected Nordic Semiconductor's devices.
Currently, these models can be run only on the `nRF54LM20B`_ device.

.. _edge_impulse_requirements_software:

Software requirements
*********************

To start working with the |EI| SDK, you must:

1. Complete :ref:`setup_sdk` (includes |NCS|, toolchain, and |EI| SDK).
#. Create an `Edge Impulse studio account <Edge Impulse studio signup_>`_ if you want to train and deploy your own machine learning models.
#. Follow the `Edge Impulse CLI installation guide`_ to install |EI| command line tools.
   They include, for example, ``edge-impulse-data-forwarder`` which can be used to forward data from a board to |EIS| for training machine learning models.

Model training
***************

This section will guide you through the complete workflow from data collection to model deployment using |EIS|.
The platform's visual interface makes it easy to experiment with different model architectures and signal processing techniques.

.. rst-class:: numbered-step

Create an account
=================

First, create a free `Edge Impulse studio account <Edge Impulse studio signup_>`_.
Your account gives you access to |EIS|, where you can manage projects, collect and label data, train models, and deploy them to your devices.
The platform provides generous free tier access, making it perfect for learning and prototyping.

.. rst-class:: numbered-step

Collect data
============

Data is the foundation of your machine learning model.
You will need representative samples that capture the patterns, events, or conditions you want your model to recognize.
|EI| makes data collection straightforward with multiple options to fit your workflow.

Choose the method that works best for your project:

* Directly from your development board - Use or modify the :ref:`data_forwarder_sample` to stream sensor data from your Nordic board.
  Use it for custom hardware setups to have full control over data collection.
  Enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option when streaming directly to |EIS| using the `Edge Impulse's data forwarder`_ CLI.
  You can also use the :ref:`data_forwarder_host_tool` to visualize and save the streamed sensor data to your local machine and than upload it to |EIS| manually.

* Quick start with Thingy:53 - If you have a Thingy:53, install the Edge Impulse - Wi-Fi firmware using the `nRF Programmer`_ mobile app, then use the `nRF Edge Impulse`_ mobile app to forward sensor data wirelessly.
  This is the fastest way to start collecting data without writing any code.

* Upload existing datasets - If you already have data, you can upload synthetic data or public datasets directly to |EIS|.
  Check `Edge Impulse Datasets`_ for community-contributed datasets you can use as a starting point.

.. tip::
   * For time-series data (sensor readings, audio), start with at least 5-10 minutes of varied data per class.
   * For image data, aim for 50-100 images per class as a starting point, with good variety in lighting, angles, and backgrounds.
   * For all data types, prioritize dataset diversity and balance the number of samples across classes to improve model performance.

For more details on data collection strategies, follow the `Edge Impulse data acquisition`_ guide.

.. rst-class:: numbered-step

Train your model
================

|EIS| guides you through creating an "Impulse", which is a pipeline that processes your raw sensor data, extracts meaningful features, and trains a neural network to recognize patterns.
The visual workflow makes it easy to experiment with different configurations and see results in real-time.

Train and deploy your model using `Edge Impulse studio`_:

* Start with :ref:`ug_edge_impulse_adding_preparing` to learn the basics of preparing and deploying your model for Nordic devices.
* Explore the comprehensive `Edge Impulse getting started guide`_ for in-depth tutorials on building different types of ML applications.

Your model is now trained and ready for deployment on Nordic devices.

Next steps
==========

* If you use Axon and need lower‑level access to the NPU beyond what |EI| provides, see :ref:`quick_start_axon_driver_model_compilation` to learn how to compile custom TensorFlow Lite models for Axon.

Application development
***********************

This section covers integration steps for a trained |EI| model into your embedded application.
The |EI| SDK provides a C++ API that makes it straightforward to run inference on your device.

.. rst-class:: numbered-step

Prepare your environment
========================

Before integrating your model, set up the |EI| development environment on your system.
This one-time setup prepares everything you need to build and deploy |EI| applications on Nordic devices.

1. :ref:`Set up Edge Impulse SDK <setup_edge_impulse>`.
#. Run the :ref:`hello_ei_sample` sample application to verify everything is working correctly.

Successfully running the :ref:`hello_ei_sample` confirms your toolchain is properly configured and ready for development.

.. rst-class:: numbered-step

Develop your application
========================

Now you can integrate your trained model into your application.
The |EI| SDK makes it easy to load your model, feed it sensor data, and get predictions with just a few API calls.

1. Add your model - Include the generated model package in your application following the instructions in :ref:`ug_edge_impulse_adding_building`.
   |EI| packages your entire inference pipeline into a portable library.

#. Implement your application logic using the |EI| SDK API:

   * See :ref:`hello_ei_sample` for a simple example showing the basic API flow from initialization to inference.
   * Explore :ref:`data_forwarder_sample` if you want to add data forwarding capabilities for continuous learning and debugging.
   * Read the `Edge Impulse C++ SDK`_ documentation for comprehensive API reference and advanced features.

.. tip::
   Start with one of the sample applications and modify it incrementally.
   This will help you understand the API structure before building your custom application from scratch.

.. rst-class:: numbered-step

Deploy your application
=======================

Build your application, flash it to your Nordic device, and verify its real-time inference on live sensor data.

.. include:: /includes/build_and_run_general.txt

Your Nordic device is now running intelligent edge AI powered by |EI|.

Next steps
**********

To work on advanced solutions, see further documentation:

* Explore advanced features - Dive deeper into the `Edge Impulse C++ SDK`_ documentation to discover advanced capabilities like anomaly detection, continuous learning, and custom processing blocks.
* Direct Axon NPU control - If you use Axon and need lower‑level access to the NPU beyond what |EI| provides, see :ref:`quick_start_axon_driver_app_development` to learn how to implement custom inference pipelines with the Axon driver API.
* Add runtime monitoring of model outputs with :ref:`nrf_edgeai_obsv_lib`.
