.. _index:

Welcome to the |EAI| for |NCS|
##############################

.. contents::
   :local:
   :depth: 2

.. note::
   This is an add-on for `nRF Connect SDK`_.

The |EAI| contains solutions and integrations for running machine learning models on Nordic Semiconductor devices.

Edge AI overview
****************

Edge AI (embedded or on-device AI) refers to running artificial intelligence and machine learning algorithms directly on embedded or resource-constrained devices, such as microcontrollers and low-power system-on-chips (SoCs).
By running models on-device you can:

* Reduce cloud connectivity and latency by making real-time decisions locally.
* Improve privacy and reliability by keeping raw sensor data on the device.
* Reduce operational cost and power usage by transmitting only compact results or events.

A typical Edge AI workflow includes collecting sensor data, extracting important features, often using digital signal processing (DSP), running the machine learning model on the device itself, and handling the results, such as sending data, triggering actions, or updating a local user interface.
To fit on small devices, these models are usually quantized and optimized for limited memory, storage, and processing power.

Key objectives
**************

The |EAI| is a practical toolkit designed to help you integrate Edge AI applications into the |NCS| ecosystem.
Its main goals are:

* Providing a compact, portable C runtime that initializes models, preparing inputs, running inference, and exposing decoded outputs.
* Delivering a set of DSP primitives, such as windowing, spectral transforms, and statistical features, which are commonly required for processing sensor data in machine learning applications.
* Including sample applications and CI-friendly examples that show how to package generated model sources, configuring Kconfig options, and building projects using `west`.

See the following documentation:

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Subpages:

   setting_up_environment.rst
   samples.rst
   integrations.rst
   libraries.rst
   release_notes.rst
   known_issues.rst
