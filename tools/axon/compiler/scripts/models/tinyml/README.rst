MLPerf™ Tiny Models Documentation
================================

Overview
--------

**MLPerf™ Tiny** is a benchmarking suite designed for evaluating deep learning models on extremely low-power systems, such as microcontrollers. This benchmark suite includes a variety of models representing real-world deep learning applications, enabling a fair comparison of the performance of different embedded devices.

All the example models in this folder are part of the **MLPerf Tiny Deep Learning Benchmarks** for Embedded Devices.

For more details, visit the `MLPerf Tiny GitHub repository <https://github.com/mlcommons/tiny/tree/master>`_.

There are four different models from the ml commons website which are supported on axon-

- `keyword_spotting : kws <kws/README.rst>`_
- `image_classification : ic <image_classification/README.rst>`_
- `visual_wake_word : vww <vww/README.rst>`_
- `anomaly_detection : ad  <anomaly_detection/README.rst>`_

Evaluation Datasets
-------------------

Each model in the benchmark suite is evaluated against specific datasets that reflect the type of data the model is designed to process. For detailed information about the datasets used for evaluation, including how the models are tested, please refer to the `Evaluation Datasets documentation <https://github.com/mlcommons/tiny/tree/master/benchmark/evaluation/datasets>`_.

Model Setup and Training
------------------------

Each model comes with a **README** file explaining the following:

- **Model Overview**: A simple description of the model.
- **Raw Datasets**: Instructions on how and where to obtain the training and testing datasets.
- **Data Pre-processing**: Guidance on how to preprocess the data to ensure it’s in the correct format for model training and evaluation.

You can find the necessary scripts to get and pre-process the dataset in the `Training Repository <https://github.com/mlcommons/tiny/tree/master/benchmark/training>`_. These resources include information needed to get started with training the models.

Setting Up the Python Environment
---------------------------------

To work with the MLPerf Tiny models, you may need to set up a Python environment and install the required Python packages using pip. Please note that instructions for setting up the environment might not be included in every individual model folder but will generally be found in the respective benchmark directories.

Notes
-----

- Setting up the Python environment and installing the necessary dependencies are vital for getting the dataset and performing the necessary functions to pre-process the dataset in the right format.
