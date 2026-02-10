.. _axon_compiler_tiny_models:

MLPerf™ Tiny Models
###################

.. contents::
   :local:
   :depth: 2

The following page describes the MLPerf™ Tiny models that are supported on Axon compiler.

Overview
********

MLPerf™ Tiny is a benchmarking suite for evaluating deep learning models on extremely low‑power systems, such as microcontrollers. 
The suite includes multiple models that represent real‑world deep learning applications and enables fair performance comparisons across embedded devices.

All example models in this folder are part of the MLPerf Tiny Deep Learning Benchmarks for Embedded Devices.
For additional background and reference material, see the `MLPerf Tiny GitHub repository`_.

The following MLPerf Tiny models from the MLCommons repository are supported on Axon:

* `Keyword spotting (KWS) <axon_compiler_kws>`_
* `Image classification (IC) <axon_compiler_image_classification>`_
* `Visual wake word (VWW) <axon_compiler_vww>`_
* `Anomaly detection (AD) <axon_compiler_anomaly_detection>`_

Evaluation datasets
*******************

Each model in the benchmark suite is evaluated using datasets that reflect the type of input data the model is designed to process.
For detailed information about the evaluation datasets and the testing methodology used by MLPerf Tiny, refer to the `Evaluation Datasets documentation`_.

Model setup and training
************************

Each model directory includes a :file:`README.rst` file that describes the following:

* Model overview, providing a high‑level description of the model.
* Raw datasets, including instructions on how and where to obtain the training and testing data.
* Data pre‑processing steps required to prepare the data for training and evaluation.

You can find scripts for downloading and pre‑processing the datasets in the `Training repository`_.
These resources provide the information needed to get started with training the models.

Setting up the Python environment
*********************************

To work with the MLPerf Tiny models, you may need to set up a `Python environment <Python_>`_.
Setting up the Python environment and installing the required dependencies is necessary for downloading datasets and performing the data pre‑processing steps in the expected format.

Available models
****************

The following sections provide detailed information about each of the supported MLPerf Tiny models, including instructions for downloading datasets, pre‑processing data, and running the compiler.

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Subpages:

   mlperf_tiny_models_anomaly_detection.rst
   mlperf_tiny_models_image_classification.rst
   mlperf_tiny_models_kws.rst
   mlperf_tiny_models_vww.rst
