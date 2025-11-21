.. _samples_nrf_edgeai:

NRF Edge AI samples
####################

This page contains an overview of the example applications that demonstrate
how to use the NRF Edge AI add-on with the nRF Connect SDK.

.. contents::
   :local:
   :depth: 2

Overview
--------

The samples demonstrate typical workflows for deploying a generated model to
an nRF device by using the `nrf_edgeai` wrapper. Each sample includes a small
set of generated model sources under ``src/nrf_edgeai_generated`` (or a
reference to how to include them).

Samples
-------

* ``regression`` - Demonstrates running a regression model (numeric prediction).
* ``classification`` - Demonstrates multi-class classification and result handling.
* ``anomaly`` - Demonstrates anomaly-detection workflows and thresholding.

Building and running
--------------------

See the sample-level build instructions in each sample folder. General build
and run notes are included here:

.. include:: ../includes/include_building_and_running_edgeai.txt

Examples
--------

Basic integration and usage examples (model init, feeding inputs and running
inference) are available in the integration guide. For quick reference the
examples are also included here:

.. include:: ../includes/include_integration_examples_edgeai.txt

