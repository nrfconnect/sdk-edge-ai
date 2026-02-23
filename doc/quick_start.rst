.. _quick_start:

Quick start
###########

.. contents::
   :local:
   :depth: 1

Welcome to nRF Edge AI quick start guides!
Ready to bring machine learning to your Nordic Semiconductor devices?
These guides will help you get started with edge AI, whether you're training your first model or deploying advanced applications.

Choose the path that best fits your project needs and experience level.
Each guide provides step-by-step instructions from setup through deployment, helping you succeed with your edge AI application.

.. tip::
   Not sure which option to choose? Check the :ref:`solution_comparison` page for a detailed comparison of features, performance, and use cases.

Basic setup variants
********************

Start here if you're new to edge AI or want to get up and running quickly.
These solutions provide high-level APIs and integrated toolchains that handle much of the complexity for you.

* :ref:`quick_start_nrf_edgeai` - Train models using Nordic Edge AI Lab and deploy them with a unified API.
  Supports both CPU execution (Neuton) and NPU acceleration (Axon) on compatible devices.
  Perfect for: Getting started quickly, standardized workflow, and the best Nordic device compatibility.
* :ref:`quick_start_edge_impulse` - Use the popular Edge Impulse platform for an end-to-end ML solution.
  Includes visual development tools, extensive documentation, and community support.
  Perfect for: Visual model development, teams already using Edge Impulse, existing Edge Impulse projects.

.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start/nrf_edgeai
   quick_start/edge_impulse

Advanced setup variants
***********************

Choose these options when you need maximum control, custom optimizations, or want to leverage specific hardware capabilities.
These approaches require more technical expertise but offer greater flexibility and performance tuning.

* :ref:`quick_start_axon_driver` - Work directly with the Axon NPU driver API for maximum control and performance.
  Compile TensorFlow Lite models and implement custom inference pipelines.
  Perfect for: Advanced optimization, custom integrations, maximum NPU performance.
* :ref:`quick_start_axon_edge_impulse` -  Combine Edge Impulse's user-friendly platform with Axon NPU hardware acceleration.
  Use Edge Impulse SDK while benefiting from NPU performance on compatible devices.
  Perfect for: Current Edge Impulse users with Axon hardware.

.. toctree::
   :maxdepth: 1
   :hidden:

   quick_start/axon_driver
   quick_start/axon_edge_impulse