.. _axon_compiler_vww:

TinyML Visual Wake Word (VWW)
#############################

.. contents::
   :local:
   :depth: 2

Overview
--------

The **Visual Wake Word (VWW)** model is designed to perform binary image classification. It detects a person in an image. The model is trained on the **COCO** dataset and can be used for visual wake word detection in real-time systems. The model is based on the **MobileNet** architecture.

The trained model can be found at the following link: `Visual Wake Word Model <https://github.com/mlcommons/tiny/tree/master/benchmark/training/visual_wake_words/trained_models>`_. 

The downloaded tflite and/or keras model files must be placed in the root directory of the model i.e. *vww/<tflite_model.tflite> or vww/<keras_model.h5>*

Raw Dataset
-----------

To download the dataset and set up the training environment for the VWW model, use the script provided at the tinyl ml commons repository below:

`download_and_train_vww.sh <https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/download_and_train_vww.sh>`_

Pre-processing
--------------

Once the dataset is downloaded, some data prep needs to be done before testing the model. This includes generating the test labels for the dataset. You can generate the test labels using the `generate_y_labels.py <https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/generate_y_labels.py>`_ script.

Running the Compiler
--------------------

The compiler executor can be run by using the provided example compiler_sample_input file. The compiler sample input expects the tflite model in the root folder of the *vww/*.

The user can run the executor by just downloading the tflite file from the location above and using the `compiler_sample_vww_input <compiler_sample_vww_input.yaml>`_ yaml file.

More advanced users who want test accuracy results must download the dataset and uncomment the test_data and test labels fields in the yaml file to be able to use the sample input yaml file.

Once the data is downloaded and pre-processed they must be kept in the directory *vww/data/*. The test data set and the labels may need to be renamed to match the input of the compiler sample input yaml file.
The name of the test data and label files are *x_test_vww.npy* and *y_test_vww.npy* respectively.

If the user decides to reference the test data files from another location, they must update the file location in the yaml file accordingly for the tinyml_vww model.

NOTE
----

- Be sure to go through the Python scripts provided in the repository to fully understand the steps for obtaining the dataset, training the model, and getting the y labels.
- If you encounter any issues during training or environment setup, refer to the scripts and ensure the correct Python packages are installed.
