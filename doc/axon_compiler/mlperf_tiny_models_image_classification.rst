.. _axon_compiler_image_classification:

TinyML Image Classification (IC)
################################

.. contents::
   :local:
   :depth: 2

Overview
--------

The **Image Classification (IC)** model is based on a **ResNet8** architecture. It is designed to classify images in the CIFAR-10 dataset, a well-known dataset for machine learning and computer vision tasks.

The trained image classification model can be downloaded from the following location:
`Image Classification Model <https://github.com/mlcommons/tiny/tree/master/benchmark/training/image_classification/trained_models>`_.

The downloaded tflite and/or keras model files must be placed in the root directory of the model i.e. *image_classification/<tflite_model.tflite> or image_classification/<keras_model.h5>*

Raw Dataset
-----------

The model is trained on the **CIFAR-10 dataset**. You can find a detailed guide on how to download the dataset and train the model in the following `README <https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/README.md>`_.

To simplify the process, you can use the script below to download the CIFAR-10 dataset and start training the model:

`download_cifar10_train_resnet.sh <https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/download_cifar10_train_resnet.sh>`_

The dataset is also available at the `link <https://www.cs.toronto.edu/~kriz/cifar.html>`_.

Data Pre-Processing
-------------------

The training script, `train.py <https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/train.py>`_, provides a detailed implementation for training the Image Classification model. You are encouraged to read through the script to understand the training and the pre-processing steps.

The folder has a `requirements.txt <https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/requirements.txt>`_ file for downloading the required Python packages and a script `prepare_training_env.sh <https://github.com/mlcommons/tiny/blob/master/benchmark/training/image_classification/prepare_training_env.sh>`_ to set up the Python environment.

Running the Compiler
--------------------

The compiler executor can be run by using the provided example compiler_sample_input file. The compiler sample input expects the tflite model in the root folder of the *image_classification/*.

The user can run the executor by just downloading the tflite file from the location above and using the `compiler_sample_ic_input <compiler_sample_ic_input.yaml>`_ yaml file.

More advanced users who want test accuracy results must download the dataset and uncomment the test_data and test labels fields in the yaml file to be able to use the sample input yaml file.

Once the data is downloaded and the necessary pre-processing steps have been performed on the data, the data files should be placed in the directory : image_classification/data/, as expected by the compiler sample input yaml file and must be renamed accordingly. The test data must be renamed *x_test_ic.npy* and the labels *y_test_ic.npy* respectively to match the input yaml file.

If the user decides to reference the test data files from another location, they must update the file location in the yaml file accordingly for the tinyml_ic model.

NOTE
----

- Be sure to go through the README and the Python scripts provided in the repository to fully understand the steps for obtaining the dataset, training the model, and performing pre-processing.
- If you encounter any issues during training or environment setup, refer to the scripts and ensure the correct Python packages are installed.
