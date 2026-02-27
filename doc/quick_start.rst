.. _quick_start:

Quick start guide
#################

.. contents::
   :local:
   :depth: 2


This section includes quick start guides for running machine learning workloads on Nordic Semiconductor devices using nRF Edge AI.
These pages cover common development paths, from simplified end‑to‑end solutions to lower‑level integrations with direct hardware control.

Select the guide that best matches your required level of control, tooling preferences, and target hardware.
Each guide walks through the required setup steps and shows how to deploy and run an edge AI application.

If you are unsure which approach fits your use case, see the :ref:`solution_comparison` page for a comparison of features, performance characteristics, and supported workflows.

Basic setup variants
********************

Use these options to get started quickly or when you prefer higher‑level tooling that abstracts most of the model deployment and runtime details.
These workflows rely on integrated toolchains and APIs to reduce setup effort:

* :ref:`quick_start_nrf_edgeai` - Use it to train models with Nordic Edge AI Lab and deploy them with a unified API.
  It supports both CPU execution (Neuton) and NPU acceleration (Axon) on compatible devices.
  Use it for easy setup, standardized workflow, and the best Nordic Semiconductor device compatibility.
* :ref:`quick_start_edge_impulse` - Use the Edge Impulse platform for an end-to-end machine learning solution.
  It includes visual development tools, extensive documentation, and community support.
  Use this option if you already work with Edge Impulse or require its tooling ecosystem.

.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start/nrf_edgeai
   quick_start/edge_impulse

Advanced setup variants
***********************

Use these options when you need lower‑level control, custom inference pipelines, or direct access to hardware acceleration features.
These workflows require more manual configuration but allow finer control over performance and resource usage.

* :ref:`quick_start_axon_driver` - Work directly with the Axon NPU driver API for maximum control, performance and low energy consumption.
  Compile TensorFlow Lite models and implement custom inference pipelines.
  Use this option for advanced optimization and direct NPU control.
* :ref:`quick_start_axon_edge_impulse` - Combine Edge Impulse's user-friendly platform with Axon NPU hardware acceleration.
  Use Edge Impulse SDK while benefiting from NPU performance on compatible devices.
  Use this option if you want to keep the Edge Impulse workflow while targeting Axon hardware.

.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start/axon_driver
   quick_start/axon_edge_impulse