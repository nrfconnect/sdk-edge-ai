.. _quick_start_nrf_edgeai:

nRF Edge AI API
###############

.. contents::
   :local:
   :depth: 2

The following guide explains how to develop and deploy machine learning applications on Nordic Semiconductor devices using the nRF Edge AI API.
This guide will walk you through bringing machine learning to your Nordic Semiconductor devices.
Whether you are training your first model or integrating AI into an existing application, you will find everything you need to get started.

The nRF Edge AI solution supports two types of models:

* `Neuton models`_ - Designed for CPU execution, compatible with a wide range of Nordic devices.
* `Axon NPU`_ models - Designed for NPU acceleration on devices equipped with the `Axon NPU`_.

After completing this guide, you will have a trained machine learning model running on your Nordic development board.

Model training
**************

This section describes how to train a machine learning model using Nordic Edge AI Lab.
It is ideal if you want to create and train custom models without implementing the full training pipeline manually. 
If you are new to machine learning, the Nordic Edge AI Lab provides a user-friendly interface that abstracts much of the complexity.

The training workflow uses the Nordic Edge AI Lab interface to configure data, select model parameters, and run training.
The tooling abstracts many of the underlying optimization details to simplify model development.

Follow the steps below to create a machine learning model.
For a detailed walkthrough of the training workflow, see the `Nordic Edge AI Lab Quick Start Guide`_.

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
You will need labeled data that represents the various conditions and scenarios your application will encounter.

You can obtain training data in one of the following ways:

* Start with preloaded datasets - Ideal for learning and experimentation. 
  `Nordic Edge AI Lab`_ includes `sample datasets <Nordic Edge AI Lab preloaded datasets_>`_  that you can use.
* Use your own data - If you have custom datasets or want to use publicly available ones, you can upload them directly to the platform.
* Collect live sensor data - Use or modify the :ref:`app_gesture_recognition` application to gather real-time data from sensors on your Nordic development board.
  See :ref:`app_gesture_recognition_data_collection` for steps on how to build the application in the data collection mode.

Refer to `Nordic Edge AI Lab Model Dataset Requirements`_ for details on data format, size requirements, and best practices for preparing your dataset.

.. note::
   Your dataset must include a ``target`` column that contains the labels or values for each data sample.
   For classification tasks, the ``target`` values must be integers starting from ``0``.

.. rst-class:: numbered-step

Train your model
================

The Nordic Edge AI Lab automates the complex process of model training, optimization, and packaging.
You will upload your dataset, configure your model parameters, and let the platform handle the processing.
The result is a highly optimized model ready to run on your Nordic device.

Train and deploy your model using `Nordic Edge AI Lab`_.
Refer to `Nordic Edge AI Lab Model Creating Pipeline`_ to see how to create and deploy a model.

.. note::
   If you want to train a model that is going to be executed on `Axon NPU`_, make sure to select the appropriate target hardware during the model creation process in `Nordic Edge AI Lab`_.

.. rst-class:: numbered-step

Next steps
==========

Once completed, you can refer to further documentation:

* Learn about `Anomaly detection <Nordic Edge AI Lab Anomaly detection_>`_ to identify unusual conditions or failures without labeled training data.
* Use `Nordic Edge AI Lab Analytics Tools`_ to analyze and improve your model's performance.
* Continue to the :ref:`quick_start_nrf_edgeai_app_dev` section to integrate your model into an application.

.. _quick_start_nrf_edgeai_app_dev:

Application development
***********************

Whether you trained your own model or are working with an existing one, this section will help you integrate it into your application.
You will set up your development environment, learn to use the nRF Edge AI API, and deploy your AI-powered application.

.. rst-class:: numbered-step

Prepare your environment
========================

Before you can start developing, you need to set up your development environment with the necessary tools and libraries.
This is a one-time setup that will enable you to build, flash, and debug AI applications on Nordic devices.

1. :ref:`Set up Edge AI library <setup_nrf_edgeai_lib>`.
#. Build and run one of the :ref:`nRF Edge AI Samples <samples_nrf_edgeai_overview>` or the :ref:`app_gesture_recognition` application to verify the setup.

.. rst-class:: numbered-step

Develop your application
========================

You must integrate the nRF Edge AI library into your application.
Complete the following steps:

1. Download your model package from `Nordic Edge AI Lab`_ after training completes.
   The package contains data required to integrate your model into your application.
   Refer to :ref:`ug_nrf_edgeai_integration` for step-by-step instructions on including the library package in your project.

#. Implement the AI functionality using :ref:`nRF Edge AI API <nrf_edgeai_lib_api>`.
   The API provides simple functions to initialize your model, run inference on input data, and retrieve predictions.

   For practical examples and code patterns, refer to:

   * The :ref:`nRF Edge AI Samples <samples_nrf_edgeai_overview>` page, that demonstrates different AI use cases.
   * The :ref:`app_gesture_recognition` application, that serves as a complete end-to-end example.


Start by running one of the sample applications to understand the API flow before customizing it for your needs.

.. rst-class:: numbered-step

Deploy your application
=======================

Build your application, flash it to your Nordic device, and verify its real-time inference on live sensor data.

.. include:: /includes/build_and_run_general.txt

.. note::
   * Axon models require a device with `Axon NPU`_ to leverage hardware acceleration.
   * Neuton models run on the CPU and are compatible with a wider range of Nordic Semiconductor devices.

Your Nordic device is now powered by edge AI, capable of making intelligent decisions locally without relying on cloud connectivity.

Next steps
**********

See further documentation:

* Read through :ref:`nrf_edgeai_lib` to understand more about the library and its capabilities.
