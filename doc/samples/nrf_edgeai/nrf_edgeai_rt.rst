.. _samples_nrf_edgeai_rt:

nRF Edge AI Runtime
###################

This page contains an overview of the example applications that demonstrate
how to use the nRF Edge AI Add-on runtime with `Nordic Edge AI Lab`_ generated models.

.. contents::
   :local:
   :depth: 2

Overview
--------

The samples demonstrate typical workflows for deploying a `Nordic Edge AI Lab`_ generated model to
an nRF device by using the `nrf_edgeai` library. Each sample includes a small
set of generated model sources under ``src/nrf_edgeai_generated`` (or a
reference to how to include them).

Samples
-------

* ``samples/nrf_edgeai/regression`` - Demonstrates running a regression model (numeric prediction).
* ``samples/nrf_edgeai/classification`` - Demonstrates multi-class classification and result handling.
* ``samples/nrf_edgeai/anomaly`` - Demonstrates anomaly-detection workflows and thresholding.

.. toctree::
   :maxdepth: 1

   rt/regression
   rt/classification
   rt/anomaly

Building and running
--------------------

See the sample-level build instructions in each sample folder. General build
and run notes are included here:

.. include:: ../../includes/include_building_and_running_edgeai.txt

Examples
--------

Basic integration and usage examples (model init, feeding inputs and running
inference) are available in the integration guide. For quick reference the
examples are also included here:

.. include:: ../../includes/include_edgeai_rt_basic_integration_example.txt
