.. _sample_hello_axon:

Hello Axon
##########

.. contents::
   :local:
   :depth: 2

The Hello Axon sample demonstrates how to run neural model inference on the Axon NPU using directly the Axon NPU driver.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The sample demonstrates how to run inference of a simple regression model on the Axon NPU.
It can be configured to use either synchronous or asynchronous inference mode.

The regression model used in the sample is taken from Zephyr's "TensorFlow Lite for Microcontrollers: Hello World" sample.
The model's task is to replicate the sine function in the range from 0 to 2π.
The TensorFlow Lite file describing this model was processed by the Axon NPU Compiler to convert it into a format accepted by the Axon NPU.
The compilation output was saved in :file:`src/generated/nrf_axon_model_hello_axon_.h`.

This sample is intended for developers working directly with the Axon NPU compiler.

Configuration
*************

|config|

Configuration options
=====================

|sample_kconfig|

.. options-from-kconfig::
   :show-type:

Building and running
********************

.. |sample path| replace:: :file:`samples/hello_axon`

.. include:: /includes/build_and_run.txt

Testing
=======

|test_sample|

#. |connect_kit|
#. |connect_terminal_kit|
#. Reset the development kit.
#. Observe the logging output.

Sample output
=============

The following output is logged in the terminal:

   .. code-block:: console

      I: Hello Axon sample
      I: Initializing Axon NPU
      I: Running asynchronous inference
      I: prediction:  0.051,  ideal  0.072
      I: prediction:  0.847,  ideal  0.842
      I: prediction: -0.491,  ideal -0.500
      ...

Dependencies
************

This sample uses the following |EAI| libraries:

* Axon NPU driver

It uses the following Zephyr libraries:

* Logging
