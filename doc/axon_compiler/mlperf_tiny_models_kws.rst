.. _axon_compiler_kws:

TinyML Keyword Spotting (KWS)
#############################

.. contents::
   :local:
   :depth: 2

This page describes a TinyML-based keyword spotting (KWS) use case for recognizing predefined keywords from audio input using a DS-CNN model.

Overview
********

The keyword spotting model is based on a Depthwise Separable Convolutional Neural Network (DS-CNN) architecture and follows the MLPerf Tiny keyword spotting reference implementation.
The model is trained on the Google Speech Commands V2 dataset.
Pre-trained model files are available in the the `MLPerf Tiny repository <Keyword spotting trained model>`_
The Axon compiler uses the exported TFLite model as input and compiles it for execution on Axon-enabled devices.
The pre-trained floating-point model is provided in TensorFlow SavedModel format, while the compiler expects a TFLite model.

Limitations and considerations
******************************

When working with this model, keep the following points in mind:

* Review :file:`README` and Python scripts in the reference repository to understand the complete workflow for dataset preparation, training, and evaluation.
* Ensure that all required Python dependencies are installed before running the training or pre-processing scripts.
* Test accuracy evaluation during compilation requires access to prepared test data and additional configuration in the compiler input file.

Running the model
*****************

You can either train the model using the reference implementation or start from a pre-trained model.
Place the downloaded TFLite model in the directory expected by the compiler input configuration file. 

Obtaining raw dataset
=====================

This model uses the Google Speech Commands V2 dataset.
The MLPerf Tiny repository includes scripts to download the dataset, train the model, and prepare test data.
Detailed instructions are provided in the `reference repository <Keyword spotting_>`_. 

Data pre-processing and model behavior
======================================

You can find all the other relevant scripts for loading and preparing the dataset in the `Keyword spotting scripts folder <Keyword spotting_>`_.
These scripts will guide you through generating the feature data required to evaluate the model and compute test accuracy.

Running the Compiler
********************

This section explains how to compile the keyword spotting model using the Axon compiler.
You run the compiler executor using a sample compiler input configuration file.
The provided sample configuration expects the TFLite model to be located in the root of the :file:`kws/`` directory.

Compiling the model without test accuracy evaluation
====================================================

Complete the following steps:

#. Download the TFLite model from the :file:`kws/` directory.
#. Use the :file:`compiler_sample_kws_input.yaml` file without modifying it.

Compiling the model with test accuracy evaluation
=================================================

Complete the following additional steps:

#. Download and pre-process CIFAR-10 dataset as described in the `reference documentation <Image classification model_>`_.
#. Uncomment the ``test_data`` and ``test_labels`` fields in the YAML file.
#. Place the processed data files in the :file:`kws/data` directory.
#. Rename the files as follows to match the sample configuration

  * :file:`x_test_kws.npy`
  * :file:`y_test_kws.npy`

  If the test data files are stored in a different location, update the file paths in the YAML configuration accordingly.

.. _axon_compiler_kws_model_script:

Experimental: KWS_MODEL_SCRIPT
******************************

.. note::
   This script is still under development and is intended for reference purposes only.

The :file:`kws_model_script.py` file is provided as an experimental example to help you develop your own keyword spotting model and data handling scripts.
The script accepts a YAML input file that defines configuration parameters for different execution modes.

An empty sample input file, :file:`kws_model_script_sample_input.yml`, is provided as a reference.

The script currently supports the following run modes:

* ``get_data``
* ``train``
* ``test``

Limitations and notes
=====================

The experimental script is intended as a reference for users who want to build custom model and dataset pipelines using the utilities provided in :file:`model_data_helper_script.py` file.

It demonstrates how to:

* Generate CSV files from raw audio data in batches
* Use the Axon feature extractor externally to generate features
* Convert feature data into NumPy format
* Convert fixed-point features to floating-point values for training
* Train and evaluate a keyword spotting model

The limitations of the current script include:

* It assumes that certain directories already exist and may fail if they are missing. 
  You must create the required directories before running the script.
* It is specific to the keyword spotting use case and is meant as an example rather than a production-ready tool.
* It does not generate Axon feature extractor executables or libraries.

Get data mode
=============

The ``get_data`` mode downloads the raw Google Speech Commands data using TensorFlow Datasets and saves it to disk.
The script can export the raw data as CSV or NumPy files, which can then be processed by the Axon feature extractor.

The configuration parameters for this mode include:

* :file:`data_directory` – Directory where the raw dataset is downloaded
* ``save_raw_data_csv`` – Save raw data as CSV files
* ``save_raw_data_npy`` – Save raw data as NumPy files
* ``train_data_fraction`` – Fraction of data used for training when generating datasets
* ``batch_file_size_limit`` – Maximum batch size, in megabytes
* ``enable_data_augmentation`` – Enable data augmentation when training with raw samples
* :file:`background_noise_dir` – Directory containing background noise samples

Train mode
==========

The ``train mode`` trains the keyword spotting model using either raw audio data or pre-generated feature data.
The training configuration includes parameters related to model definition, feature generation, and training behavior, such as:

* ``model_name`` – Unique name for the model
* :file:`model_directory` – Directory containing a pre-trained or partially trained model
* ``use_raw_data`` – Enable training directly from raw audio samples
* ``feature_type`` – Feature type (mfcc or axon_mfcc)
* :file:`axon_fe_dll_path` – Path to the Axon feature extractor library (required for axon_mfcc)
* :file:`train/val/test_data` – Paths to feature data files
* :file:`train/val/test_label` – Paths to label files
* ``sampling_rate`` – Audio sampling rate
* ``audio_duration_ms`` – Duration of audio samples
* ``window_size_ms`` – Window size for feature extraction
* ``window_stride_ms`` – Window stride for feature extraction
* ``dct_coefficient_count`` – Number of DCT coefficients
* ``labels_count`` – Number of output labels
* ``learning_rate`` – Training learning rate
* ``model_training_epochs`` – Number of training epochs
* ``batch_size`` – Training batch size
* ``mfcc_shift`` – Fixed-point radix used for Axon-generated features

This list is not exhaustive. 
The YAML file may contain additional fields that are included for demonstration purposes and are not used by the script.

Test mode
=========

The ``test mode`` evaluates a trained model using prepared test data.
The configuration parameters include:

* :file:`model_directory` – Directory containing the trained model
* :file:`test_feature_data` – Path to the NumPy test data file
* :file:`test_feature_label` – Path to the NumPy test label file
