.. _axon_compiler_image_classification:

TinyML Image Classification (IC)
################################

.. contents::
   :local:
   :depth: 2

This page describes a TinyML-based image classification use case using a ResNet8 model trained on the CIFAR-10 dataset.

Overview
********

The image classification model is based on a ResNet8 architecture and follows the MLPerf Tiny image classification reference implementation.
It is designed to classify images from the CIFAR-10 dataset, which is commonly used for benchmarking machine learning and computer vision workloads.
The Axon compiler uses the exported TFLite model as input and compiles it for execution on Axon-enabled devices.

Limitations and considerations
******************************

When working with this model, keep the following points in mind:

* Review :file:`README` and Python scripts in the reference repository to understand the complete workflow for dataset preparation, training, and evaluation.
* Ensure that all required Python dependencies are installed before running the training or pre-processing scripts.
* Test accuracy reporting requires access to the CIFAR-10 test dataset and additional configuration in the compiler input file.

Running the model
*****************

To start working with this model, download the trained image classification model from the `MLPerf Tiny repository <Image classification model_>`_.
You must place the download TFLite model and Keras model files in the root directory of the model (:file:`image_classification/<tflite_model.tflite>` or :file:`image_classification/<keras_model.h5>`).

Obtaining raw dataset
=====================

The model is trained on the CIFAR-10 dataset. 
To simplify the process, you can use the :file:`download_cifar10_train_resnet.sh` script in the `image classification folder structure <Image classification model_>`_ to download the CIFAR-10 dataset and start training the model.
Alternatively, you can obtain it from the `CIFAR dataset`_ page.

Data pre-processing and model behavior
======================================

The training and data pre-processing steps are implemented in the reference training script :file:`train.py` in the `MLPerf Tiny image classification training repository <Image classification model_>`_.

Reviewing this script helps clarify how the CIFAR-10 images are processed and how the ResNet8 model is trained.

The repository also includes:

* A :file:`requirements.txt` file that lists the required Python packages.
* A :file:`prepare_training_env.sh` script for setting up the Python environment.

Running the Compiler
********************

This section explains how to compile the image classification model using the Axon compiler.
You can run the compiler executor using a sample compiler input configuration file.
The provided sample configuration expects the TFLite model to be located in the root of the :file:`image_classification/` directory.

Compiling the model without test accuracy evaluation
====================================================

Complete the following steps:

#. Download the TFLite model from the :file:`image_classification/` directory.
#. Use the :file:`compiler_sample_ic_input.yaml` file without modifying it.

Compiling the model with test accuracy evaluation
=================================================

Complete the following additional steps:

#. Download and pre-process CIFAR-10 dataset as described in the `reference documentation <Image classification model_>`_.
#. Uncomment the ``test_data`` and ``test_labels`` fields in the YAML file.
#. Place the processed data files in the :file:`image_classification/data` directory.
#. Rename the files as follows to match the sample configuration

  * :file:`x_test_ic.npy`
  * :file:`y_test_ic.npy`

  If the test data files are stored in a different location, update the file paths in the YAML configuration accordingly.
