.. _axon_compiler_anomaly_detection:

TinyML Anomaly Detection
########################

.. contents::
   :local:
   :depth: 2

This model demonstrates a `TinyML-based anomaly detection <MLPerf Tiny anomaly detection_>`_ use case for identifying abnormal machine sounds.

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

Download a pre-trained model and compile it with Axon.
You do not need to download the dataset, pre-process data, or train the model for that workflow.
Complete the following steps:

#. Download a TFLite or Keras model from the `Pre-trained anomaly detection model`_.
#. Place the file in the directory expected by the compiler input configuration:

   .. code-block:: text

      anomaly_detection/<model.tflite>
      anomaly_detection/<model.h5>

Obtaining raw dataset (optional)
================================

You only need to obtain the raw dataset if you plan to train or retrain the model yourself.
To download the raw dataset, run the :file:`get_dataset.sh` script in the `reference repository <Anomaly detection script_>`_.

Data pre-processing and model behavior (optional)
=================================================

You only need to pre-process the data for training, retraining, or test accuracy evaluation during compilation.
For the required pre-processing steps, see the `reference repository <Anomaly detection training_>`_.
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
