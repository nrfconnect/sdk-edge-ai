.. _quick_start_nrf_edgeai:

nRF Edge AI API - Quick Start Guide
###################################

.. contents::
   :local:
   :depth: 2

Welcome to the nRF Edge AI quick start guide!
This guide will walk you through the complete journey of bringing machine learning to your Nordic Semiconductor devices.
Whether you're training your first model or integrating AI into an existing application, you'll find everything you need to get started.

The nRF Edge AI solution supports two types of models:

* `Neuton models`_ - designed for CPU execution, compatible with a wide range of Nordic devices.
* `Axon NPU`_ models - designed for NPU acceleration on devices equipped with the `Axon NPU`_.

By the end of this guide, you'll have a trained ML model running on your Nordic development board!

Model training
**************

If you are a data scientist or machine learning engineer and want to train your own ML model, this section is for you!
Don't worry if you're new to machine learning - we've designed the process to be straightforward and accessible.
The Nordic Edge AI Lab provides a user-friendly interface that handles much of the complexity for you, so you can focus on your application rather than the intricacies of ML optimization.

Follow these steps to create your first ML model:

.. rst-class:: numbered-step

Create an account
=================

Create your `Nordic Edge AI Lab`_ account.
This web tooling will enable you to generate a model package that is compatible with the |EAILib| API.

.. rst-class:: numbered-step

Collect training data
=====================

Data is the foundation of any machine learning model.
The quality and quantity of your training data directly impact how well your model will perform in real-world scenarios.
You'll need labeled data that represents the various conditions and scenarios your application will encounter.

You can obtain training data in one of the following ways:

* **Start with preloaded datasets** - Perfect for learning and experimentation, `Nordic Edge AI Lab`_ includes sample datasets you can use right away (see `Nordic Edge AI Lab Model Dataset Uploading`_).
* **Use your own data** - If you have custom datasets or want to use publicly available ones, you can upload them directly to the platform.
* **Collect live sensor data** - Use or modify the :ref:`app_gesture_recognition` application to gather real-time data from sensors on your Nordic development board.
  Check :ref:`app_gesture_recognition_data_collection` for instructions on how to build the application in the data collection mode.

.. tip::
   Don't have data yet? Start with the preloaded datasets to familiarize yourself with the platform, then collect your own data when you're ready.

Refer to `Nordic Edge AI Lab Model Dataset Requirements`_ for details on data format, size requirements, and best practices for preparing your dataset.

.. rst-class:: numbered-step

Train your model
================

Now comes the exciting part - training your model!
The Nordic Edge AI Lab automates the complex process of model training, optimization, and packaging.
You'll upload your dataset, configure your model parameters, and let the platform handle the heavy lifting.
The result is a highly optimized model ready to run on your Nordic device.

Train and deploy your model using `Nordic Edge AI Lab`_.
Refer to `Nordic Edge AI Lab Model Creating Pipeline`_ to find out about the steps involved in creating and deploying a model.

.. note::
   If you want to train a model that is going to be executed on `Axon NPU`_, make sure to select the appropriate target hardware during the model creation process in `Nordic Edge AI Lab`_.

.. rst-class:: numbered-step

Next steps
==========

Congratulations! You now have a trained model ready for deployment.

* Explore `Nordic Edge AI Lab Documentation`_ to learn more about advanced features, model optimization techniques, and best practices for training and deployment.
* Continue to the Application Development section below to integrate your model into an application.

Application development
***********************

Whether you trained your own model or are working with an existing one, this section will help you integrate it into your application.
You'll set up your development environment, learn to use the nRF Edge AI API, and deploy your first AI-powered application.

.. rst-class:: numbered-step

Prepare your environment
========================

Before you can start developing, you need to set up your development environment with the necessary tools and libraries.
This is a one-time setup that will enable you to build, flash, and debug AI applications on Nordic devices.

1. :ref:`Set up Edge AI library <setup_nrf_edgeai_lib>`.
#. Build and run one of the :ref:`Edge AI Samples <samples_nrf_edgeai_overview>` or the :ref:`app_gesture_recognition` application to verify the setup.

.. rst-class:: numbered-step

Develop your application
========================

With your environment set up and model ready, it's time to bring them together!
Integrating the nRF Edge AI library into your application is straightforward thanks to the unified API, no matter if you're using Neuton or Axon models.

Check out the following steps to get started:

1. **Download your model package** from `Nordic Edge AI Lab`_ after training completes.
   The package contains everything needed to integrate your model into your application.
   Refer to :ref:`ug_nrf_edgeai_integration` for step-by-step instructions on including the library package in your project.

#. **Implement the AI functionality** using :ref:`nRF Edge AI API <nrf_edgeai_lib_api>`.
   The API provides simple functions to initialize your model, run inference on input data, and retrieve predictions.

   For practical examples and code patterns, refer to:

   * The :ref:`Edge AI Samples <samples_nrf_edgeai_overview>` - demonstrate different AI use cases
   * The :ref:`app_gesture_recognition` application - a complete end-to-end example

.. tip::
   Start by running one of the sample applications to understand the API flow before customizing it for your needs.

.. rst-class:: numbered-step

Deploy your application
=======================

The moment of truth - time to see your AI model running on actual hardware!
Build your application, flash it to your Nordic device, and watch as it performs real-time inference on live sensor data.

.. include:: /includes/build_and_run_general.txt

.. important::
   * **Axon models** require a device with `Axon NPU`_ to leverage hardware acceleration.
   * **Neuton models** run on the CPU and are compatible with a wider range of Nordic Semiconductor devices.

Congratulations! You've now completed the full cycle from model training to deployment.
Your Nordic device is now powered by edge AI, capable of making intelligent decisions locally without relying on cloud connectivity.

.. rst-class:: numbered-step

Next steps
==========

* Check out :ref:`nrf_edgeai_lib` to understand more about the library and its capabilities.
