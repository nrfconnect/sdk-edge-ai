.. _axon_compiler_anomaly_detection:

TinyML Anomaly Detection
########################

.. contents::
   :local:
   :depth: 2

This model demonstrates a TinyML-based anomaly detection use case for identifying abnormal machine sounds. 

Overview
********

The model is based on a deep autoencoder architecture and follows the MLPerf Tiny anomaly detection reference implementation.
The reference implementation provides step-by-step instructions for:

* Downloading the dataset
* Training the model
* Converting the trained model to TFLite format
* Testing the model using the TFLite runtime

Limitations and considerations
******************************

When working with this model, keep the following points in mind:

* Review :file:`README` and Python scripts in the reference repository to understand the complete workflow for dataset preparation, training, and evaluation.
* Ensure that all required Python dependencies are installed before running the training or pre-processing scripts.
* Keep in mind, that test accuracy metrics are not generated for this model because it is not a classification model.
  Additionally, only classification models are currently supported for producing test accuracy reports with Axon.

Running the model
*****************

For detailed instructions, see the `MLPerf Tiny anomaly detection` page.
You can also refer to a `Pre-trained anomaly detection model`_.

Place the downloaded TFLite or Keras model in the directory expected by the compiler input configuration file, for example:

.. code-block:: text

   anomaly_detection/<model.tflite>
   anomaly_detection/<model.h5>

Obtainig raw dataset
====================

This section describes how to obtain the raw dataset used for training and evaluation.
Download it by running the :file:`get_dataset.sh` script provided in the `reference repository <anomaly detection script_>`_.

Data pre-processing and model behavior
======================================

This section summarizes the required data pre-processing steps and explains the model output behavior.

You can find data pre-processing steps required for training and testing in the `reference repository <Anomaly detection training_>`_.
These steps convert the raw audio data into the format expected by the anomaly detection model.

The model output is an anomaly score derived from the reconstruction error.
During testing, the model computes the Root Mean Square (RMS) error between the original input and the reconstructed output.
This RMS value is used to determine whether a given input represents anomalous behavior.

To understand how anomaly scores are computed and interpreted, see the `Anomaly score calculation script`_.

Running the compiler
********************

This section explains how to compile the anomaly detection model for Axon.
You can run the compiler executor using the provided sample compiler input configuration file.
The sample configuration expects the TFLite model to be located in the root of the :file:`anomaly_detection/` directory.

Compiling the model without test accuracy evaluation
====================================================

Complete the following steps:

#. Download the TFLite model from the :file:`anomaly_detection/` directory.
#. Use the :file:`compiler_sample_ad_input.yaml` file without modifying it.

Compiling the model with test accuracy evaluation
=================================================

Complete the following additional steps:

#. Download and pre-process the dataset as described in the `reference repository documentation <Anomaly detection training_>`_.
#. Uncomment the ``test_data`` and ``test_labels`` fields in the YAML file.
#. Place the processed data files in the :file:`anomaly_detection/data` directory.
#. Rename the files as follows to match the sample configuration

  * :file:`x_test_ad.npy`
  * :file:`y_test_ad.npy`

  If the test data files are stored in a different location, update the file paths in the YAML configuration accordingly.
