.. _nrf_machine_learning_app_model_runners:

nRF Machine Learning: Model runners
###################################

.. contents::
   :local:
   :depth: 2

The nRF Machine Learning application provides a set of machine learning model runners.
Each runner provides a common interface for running the machine learning model with a different backend.

Available model runners
***********************

Edge Impulse (using EI wrapper)
   This runner is for models created by the `Edge Impulse studio`_.
   Set the :kconfig:option:`CONFIG_ML_APP_ML_RUNNER_EI` Kconfig option to enable this runner.
   It uses the :ref:`ei_wrapper` to handle data buffering and running machine learning model inference in a separate thread.
   Edge Impulse wrapper provides Kconfig options for your own configuration.

Neuton
   This runner is for models created by the `Neuton platform`_.
   Set the :kconfig:option:`CONFIG_ML_APP_ML_RUNNER_NEUTON` Kconfig option to enable this runner.
   It runs the machine learning model inference in a separate low priority thread.
   The Neuton solution is responsible for data buffering and handles it internally.
   Place the model archive file in the :file:`applications/machine_learning/models/` directory and use the :kconfig:option:`CONFIG_ML_APP_ML_RUNNER_NEUTON_ARCHIVE_NAME` Kconfig option for the configuration.

Stub
   The stub runner is used as baseline for model and framework profiling and it is not running any actual model.
   Set the :kconfig:option:`CONFIG_ML_APP_ML_RUNNER_STUB` Kconfig option to enable this runner.
   It uses a separate thread that submits an empty ``ml_result_event`` after receiving enough data.
   You can set the number of received frames of data and values in a single frame using Kconfig options.
   You can also set the thread stack size and priority using Kconfig options to match those of other runners.
   The stub acts like an actual runner, including system overhead, but excluding running a machine learning model inference and framework library operations.
