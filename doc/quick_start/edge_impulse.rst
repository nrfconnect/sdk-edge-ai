.. _quick_start_edge_impulse:

Edge Impulse - Quick Start Guide
################################

.. contents::
   :local:
   :depth: 2

Welcome to the |EI| quick start guide!
This guide will help you bring the power of |EI| machine learning to your Nordic Semiconductor devices.
|EI| offers an intuitive, end-to-end platform for developing ML models optimized for embedded systems.

Whether you're collecting sensor data, training your first ML model, or deploying AI at the edge, this guide will walk you through each step of the journey.
By the end, you'll have a working ML application running on your Nordic device!

Model training
***************

Ready to train your own machine learning model?
This section guides you through the complete workflow from data collection to model deployment using |EIS|.
The platform's visual interface makes it easy to experiment with different model architectures and signal processing techniques, even if you're new to embedded machine learning.

.. rst-class:: numbered-step

Create an account
=================

First, create your free `Edge Impulse studio account <Edge Impulse studio signup_>`_.
Your account gives you access to |EIS|, where you'll manage projects, collect and label data, train models, and deploy them to your devices.
The platform provides generous free tier access, making it perfect for learning and prototyping.

.. rst-class:: numbered-step

Collect data
============

Data is the foundation of your machine learning model.
You'll need representative samples that capture the patterns, events, or conditions you want your model to recognize.
|EI| makes data collection straightforward with multiple options to fit your workflow.

Choose the method that works best for your project:

* **Direct from your development board** - Use or modify the :ref:`ei_data_forwarder_sample` to stream sensor data directly from your Nordic board to |EIS|.
  This is ideal for custom hardware setups and gives you full control over data collection.

* **Quick start with Thingy:53** - If you have a Thingy:53, install the ``Edge Impulse - Wi-Fi`` firmware using the `nRF Programmer`_ mobile app, then use the `nRF Edge Impulse` mobile app to forward sensor data wirelessly.
  This is the fastest way to start collecting data without writing any code!

* **Upload existing datasets** - Already have data? Upload synthetic data or public datasets directly to |EIS|.
  Check `Edge Impulse Datasets`_ for community-contributed datasets you can use as a starting point.

.. tip::
   * For **time-series data** (sensor readings, audio): Start with at least 5-10 minutes of varied data per class.
   * For **image data**: Aim for 50-100 images per class as a starting point, with good variety in lighting, angles, and backgrounds.
   * For all types: More diverse data leads to better model performance.
     Balance your dataset across all classes.

For more details on data collection strategies, follow the `Edge Impulse data acquisition` guide.

.. rst-class:: numbered-step

Train your model
================

With your data collected, it's time to build your ML model!
|EIS| guides you through creating an "Impulse" - a pipeline that processes your raw sensor data, extracts meaningful features, and trains a neural network to recognize patterns.
The visual workflow makes it easy to experiment with different configurations and see results in real-time.

Train and deploy your model using `Edge Impulse studio`_:

* Start with :ref:`ug_edge_impulse_adding_preparing` to learn the basics of preparing and deploying your model for Nordic devices.
* Explore the comprehensive `Edge Impulse getting started guide`_ for in-depth tutorials on building different types of ML applications.

Congratulations! Your model is now trained and ready for deployment on Nordic devices.

Application development
***********************

Ready to bring your |EI| model to life on Nordic hardware?
This section covers everything you need to integrate a trained |EI| model into your embedded application.
The |EI| SDK provides a C++ API that makes it straightforward to run inference on your device.

.. rst-class:: numbered-step

Prepare your environment
========================

Before integrating your model, set up the |EI| development environment on your system.
This one-time setup prepares everything you need to build and deploy |EI| applications on Nordic devices.

1. :ref:`Set up Edge Impulse SDK <setup_edge_impulse>`.
#. Run the :ref:`hello_ei_sample` sample application to verify everything is working correctly.

.. tip::
   Successfully running the :ref:`hello_ei_sample` confirms your toolchain is properly configured and ready for development.

.. rst-class:: numbered-step

Develop your application
========================

Now for the exciting part - integrating your trained model into a real application!
The |EI| SDK makes it easy to load your model, feed it sensor data, and get predictions with just a few API calls.

1. **Add your model** - Include the generated model package in your application following the instructions in :ref:`ug_edge_impulse_adding_building`.
   |EI| packages your entire inference pipeline into a portable library.

#. **Implement your application logic** using the |EI| SDK API:

   * Check out :ref:`hello_ei_sample` for a simple example showing the basic API flow from initialization to inference.
   * Explore :ref:`ei_data_forwarder_sample` if you want to add data forwarding capabilities for continuous learning and debugging.
   * Dive into the `Edge Impulse C++ SDK`_ documentation for comprehensive API reference and advanced features.

.. tip::
   Start with one of the sample applications and modify it incrementally. This helps you understand the API structure before building your custom application from scratch.

.. rst-class:: numbered-step

Deploy your application
=======================

The moment of truth - time to see your AI model running on actual hardware!
Build your application, flash it to your Nordic device, and watch as it performs real-time inference on live sensor data.

.. include:: /includes/build_and_run_general.txt

Congratulations! Your Nordic device is now running intelligent edge AI powered by |EI|.
You've successfully completed the journey from data collection to deployment!

.. rst-class:: numbered-step

Next steps
==========

Ready to take your application to the next level?

* **Accelerate with Axon NPU** - If you have a device with `Axon NPU`_, check out :ref:`quick_start_axon_edge_impulse` to learn how to combine |EI| with Axon hardware acceleration for significantly faster inference times.
* **Explore advanced features** - Dive deeper into the `Edge Impulse C++ SDK`_ documentation to discover advanced capabilities like anomaly detection, continuous learning, and custom processing blocks.
