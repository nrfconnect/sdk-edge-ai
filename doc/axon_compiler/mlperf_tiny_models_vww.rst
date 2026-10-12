.. _axon_compiler_vww:

TinyML Visual Wake Word (VWW)
#############################

.. contents::
   :local:
   :depth: 2


This page describes a `TinyML-based visual wake word (VWW) <Visual wake word_>`_ use case for detecting the presence of a person in an image using a MobileNet-based model.

Overview
********

The visual wake word model performs binary image classification to determine whether a person is present in an input image.
It is based on a MobileNet architecture and follows the MLPerf Tiny visual wake word reference implementation.

The model is trained on the COCO dataset and is suitable for low-power, real-time visual wake word detection.
Pre-trained model files are available from the `MLPerf Tiny repository <Visual wake word trained model_>`_
The Axon compiler uses the exported TFLite model as input and compiles it for execution on Axon-enabled devices.

Limitations and considerations
******************************

When working with this model, keep the following points in mind:

* Review the Python scripts provided in the reference repository to understand the full workflow for dataset download, training, and label generation.
* Ensure that all required Python dependencies are installed before running training or data pre-processing scripts.
* Test accuracy evaluation during compilation requires prepared test data and label files, as well as additional configuration in the compiler input file.

Running the model
*****************

Download a pre-trained model and compile it with Axon.
You do not need to download the dataset, pre-process data, or train the model for that workflow.
Complete the following steps to run the model:

#. Download the TFLite or Keras model from the `MLPerf Tiny repository <Visual wake word trained model_>`_.
#. Place the file in the directory expected by the compiler input configuration (:file:`vww/<model.tflite>` or :file:`vww/<model.h5>`).

Obtaining raw dataset (optional)
================================

You only need to obtain the raw dataset if you plan to train or retrain the model yourself.
The visual wake word model is trained on data derived from the COCO dataset.
Use the :file:`download_and_train_vww.sh` script in the `MLPerf Tiny repository <Visual wake word_>`_ to download the dataset and set up the training environment.

Data pre-processing and model behavior (optional)
=================================================

You only need to pre-process the data for training, retraining, or test accuracy evaluation during compilation.
After downloading the dataset, run the :file:`generate_y_labels.py` script to generate the test label files for evaluation.

Running the compiler
********************

This section explains how to compile the visual wake word model using the Axon compiler.

You run the compiler executor using a sample compiler input configuration file.
The provided sample configuration expects the TFLite model to be located in the root of the :file:`vww/` directory.

Compiling the model without test accuracy evaluation
====================================================

Complete the following steps:

#. Download the TFLite model from the :file:`vww/` directory.
#. Use the :file:`compiler_sample_vww_input.yaml` file without modifying it.

Compiling the model with test accuracy evaluation
=================================================

Complete the following additional steps:

#. Download and pre-process CIFAR-10 dataset as described in the `reference documentation <Image classification model_>`_.
#. Uncomment the ``test_data`` and ``test_labels`` fields in the YAML file.
#. Place the processed data files in the :file:`vww/data` directory.
#. Rename the files as follows to match the sample configuration

  * :file:`x_test_vww.npy`
  * :file:`y_test_vww.npy`

  If the test data files are stored in a different location, update the file paths in the YAML configuration accordingly.
