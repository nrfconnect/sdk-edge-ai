.. _quick_start_edge_impulse:

Edge Impulse
############

.. contents::
   :local:
   :depth: 2

The following guide explains how to develop and deploy machine learning applications on Nordic Semiconductor devices using |EI|.
It is ideal if you want an end‑to‑end workflow for data collection, model training, and deployment on embedded targets.

To follow this guide, you should be familiar with basic embedded systems development.
The guide covers the steps required to collect data, train a model using Edge Impulse tools, and deploy the resulting model to a Nordic device.

After completing this guide, you will have a machine learning application running on a Nordic Semiconductor device using |EI|.

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

* Directly from your development board - Use or modify the :ref:`ei_data_forwarder_sample` to stream sensor data directly from your Nordic board to |EIS|.
  Use it for custom hardware setups to have full control over data collection.

* Quick start with Thingy:53 - If you have a Thingy:53, install the Edge Impulse - Wi-Fi firmware using the `nRF Programmer`_ mobile app, then use the `nRF Edge Impulse` mobile app to forward sensor data wirelessly.
  This is the fastest way to start collecting data without writing any code.

* Upload existing datasets - If you already have data, you can upload synthetic data or public datasets directly to |EIS|.
  Check `Edge Impulse Datasets`_ for community-contributed datasets you can use as a starting point.

.. tip::
   * For time-series data (sensor readings, audio), start with at least 5-10 minutes of varied data per class.
   * For image data, aim for 50-100 images per class as a starting point, with good variety in lighting, angles, and backgrounds.
   * For all data types, prioritize dataset diversity and balance the number of samples across classes to improve model performance.

For more details on data collection strategies, follow the `Edge Impulse data acquisition` guide.

.. rst-class:: numbered-step

Train your model
================

|EIS| guides you through creating an "Impulse", which is a pipeline that processes your raw sensor data, extracts meaningful features, and trains a neural network to recognize patterns.
The visual workflow makes it easy to experiment with different configurations and see results in real-time.

Train and deploy your model using `Edge Impulse studio`_:

* Start with :ref:`ug_edge_impulse_adding_preparing` to learn the basics of preparing and deploying your model for Nordic devices.
* Explore the comprehensive `Edge Impulse getting started guide`_ for in-depth tutorials on building different types of ML applications.

Your model is now trained and ready for deployment on Nordic devices.

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
   * Explore :ref:`ei_data_forwarder_sample` if you want to add data forwarding capabilities for continuous learning and debugging.
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

* Accelerate with Axon NPU - If you have a device with `Axon NPU`_, see :ref:`quick_start_axon_edge_impulse` to learn how to combine |EI| with Axon hardware acceleration for significantly faster inference times.
* Explore advanced features - Dive deeper into the `Edge Impulse C++ SDK`_ documentation to discover advanced capabilities like anomaly detection, continuous learning, and custom processing blocks.
