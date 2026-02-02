.. _axon_compiler_kws:

TinyML Keyword Spotting (KWS)
#############################

.. contents::
   :local:
   :depth: 2

Overview
--------

The **Keyword Spotting (KWS)** model is designed to recognize specific keywords from audio input. The model is trained on the **Google Speech Commands V2 dataset**. The model is a **DS-CNN** based model.

The trained model file can be found at the following location: `Key Word Spotting <https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting/trained_models>`_

The downloaded tflite model should be placed in the root folder i.e. *kws/<tflite_model.tflite>* where the yaml file expects it. The pre-trained floating point model is saved in TensorFlow's SavedModel format.

Raw Dataset
-----------

To test the model and evaluate its performance, you'll need test data. The **TinyML repository** includes code to load the **Google Speech Commands V2 dataset** and to prepare the test data. The information to get the dataset, train the model and perform the necessary pre-processing on it can be found in the `README <https://github.com/mlcommons/tiny/blob/master/benchmark/training/keyword_spotting/README.md>`_.

Data Pre-processing
-------------------

You can find all the other relevant scripts for loading and preparing the dataset in the following directory:

`Keyword Spotting Scripts <https://github.com/mlcommons/tiny/tree/master/benchmark/training/keyword_spotting>`_

These scripts will guide you through the process of obtaining the necessary test data and performing the necessary pre-processing for evaluating the model's accuracy.

Running the Compiler
--------------------

The compiler executor can be run by using the provided example compiler_sample_input file. The compiler sample input expects the tflite model in the root folder of the *kws/*.

The user can run the executor by just downloading the tflite file from the location above and using the `compiler_sample_kws_input <compiler_sample_kws_input.yaml>`_ yaml file.

More advanced users who want test accuracy results must download the dataset and uncomment the test_data and test labels fields in the yaml file to be able to use the sample input yaml file.

Once the data is downloaded and the necessary pre-processing steps have been performed on the data, the data files should be placed in the directory : kws/data/, as expected by the compiler sample input yaml file and must be renamed accordingly. The test data must be renamed *x_test_kws.npy* and the labels *y_test_kws.npy* respectively to match the input yaml file.

If the user decides to reference the test data files from another location, they must update the file location in the yaml file accordingly for the tinyml_kws model.

NOTES
-----

- Be sure to go through the `README` and the Python scripts provided in the repository to fully understand the steps for obtaining the dataset, training the model, and performing pre-processing.
- If you encounter any issues during training or environment setup, refer to the scripts and ensure the correct Python packages are installed.
- Ensure that you have the correct dataset and test data to evaluate the model's performance accurately.

** *EXPERIMENTAL* ** : KWS_MODEL_SCRIPT 
---------------------------------------

The *kws_model_script.py* is shared as an example for users to generate their own model scripts as needed.

The script takes in an YAML input file with config parameters in them for the different script run modes.

An empty sample input yaml file `<kws_model_script_sample_input.yml>`_ is provided for the user as a reference as to how the yaml input to the *kws_model_script.py* looks like. 

The different script run modes are as follows:

- train
- test
- get_data

NOTE : This script is still under development and it is important to mention a few caveats involved in trying to run this script.

- The script assumes certain directories to be present by default and may generate errors when those directories are not found. The user is advised to create directories as needed.
- The script strictly tries to give an example for the kws model and is presented as a reference to the user to develop their own model and data handling scripts.
- The script does not generate axon feature extractor exe/dll and the explanation for being able to generate the exe and the dll is not part of this document.

Get Data
--------
The `get_data` mode is used to perform the steps needed to get the data from a source. This will download the raw speech commands data from the tensorflow_datasets and save them as a csv files.

The csv files can be used by the axon feature extractor exe to generate features using axon.

The config parameters for the getting data are as follows - 

- *data_directory* : Directory where the raw data can be downloaded
- *save_raw_data_csv* : saves the raw data as a csv when calling the script in get_data mode.
- *save_raw_data_npy* : saves the raw data as a numpy when calling the script in get_data mode.
- *train_data_fraction* : the fraction of test and validation data used when generating the csv or when training the model using raw data
- *batch_file_size_limit* : the maximum sample size of the batches in mb
- *enable_data_augmentation* : enables data augmentation when training with raw samples
- *background_noise_dir* : directory to supply in background noise when training with raw samples

Train
------
This mode trains the model based on the features provided. 

The train model config has different params related to the training of a KWS model and can be modified to provide extra meta information about getting the features from the raw dataset.

The parameters in the train_model_config are as follows  - 

- *model_name* : unique name for the model files. 
- *model_directory* : the directory of the model. user must provide a model pre-trained or a partially trained model to start training.
- *use_raw_data* : this flag enables training using the raw data directly. 
- *feature_type* : "mfcc" or "axon_mfcc", tells the script to either generate tensorflow mfccs or generate mfccs using an axon dll
- *axon_fe_dll_path* : this field must be defined if the user wants to generate mfccs using the axon feature extractor and the feature type is "axon_mfcc"
- *train/val/test_data* : paths to the csv directories containing the features generated by axon fe exe or the numpy files of the features
- *train/val/test_label* : path the numpy files of the labels for the train, val and test data sets
- *sampling_rate* : sampling rate of the audio samples
- *audio_duration_ms* : audio_duration of individual raw audio samples
- *window_size_ms* : size of the window on the raw audio samples
- *window_stride_ms* : size of the stride on the raw audio samples for the windowing function
- *dct_coefficient_count* : the dct coefficient count
- *labels_count* : the count of the labels in the data sets
- *learning_rate* : learning rate for training the models
- *model_training_epochs* : the number of epochs for training the model
- *batch_size* : the batch size to train the models
- *mfcc_shift* : the radix of the feature data generated using axon feature extraction

NOTE : This is not an exhaustive list of the parameters a model/data script will need, this is a very basic example of an input yaml file to a user designed script. The yaml file may contain fields that are not being used in the script and are just present for demonstration as a field in the yaml file.

Test
----
This mode simply takes in the test vector and the trained model and performs evaluation of the model on the test data. 

The parameters are as follows - 

- *model_directory* : the directory of the model that the user plans on testing
- *test_feature_data* : the directory of the numpy test data file 
- *test_feature_label* : the directory of the numpy test label file

NOTE:
-----

The above modes are not the extensive list of modes available to the user. The user can design a script based on their requirements. 

This script simply acts as an example to the user who intends to use the methods in the `model_data_helper_script.py <../../../utility/model_data_helper_script.py>`_ for their own model and dataset.

The script shows how the user can use the raw data to generate a CSV file in batches and then use the `axon_fe` exe to generate the features on Axon out of band, using some command line functions, and generate feature output CSV files. 

Once the CSV files are generated out of band, the user can use the methods in the *model_data_helper_script* to generate the numpy files, convert them from a fixed radix to floating point for training purposes, and get a trained model.