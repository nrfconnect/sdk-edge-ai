.. _index:

Welcome to the |EAI| for |NCS|
#######################################

This is add-on for `nRF Connect SDK`_.

The |EAI| contains solutions and integrations for running machine learning models on Nordic Semiconductor devices.

Edge AI overview
################

Edge AI (embedded or on-device AI) brings machine learning inference to constrained devices such as microcontrollers and low-power SoCs. By running models on-device you can:

* Reduce cloud connectivity and latency by making real-time decisions locally.
* Improve privacy and reliability by keeping raw sensor data on the device.
* Reduce operational cost and power usage by transmitting only compact results or events.

Typical Edge AI workflows include data collection, feature extraction (DSP), on-device inference and result handling (telemetry, actuation, or local UI). Models are often quantized and optimized for limited RAM/flash and CPU/FPU capabilities.

Add-on goals
############

The nRF Edge AI add-on provides an opinionated, production-oriented toolkit for integrating on-device models into the nRF Connect SDK ecosystem. The primary goals are:

* Provide a compact, portable C runtime (the `Nordic Edge AI Lab`_ runtime) that initializes models, prepares inputs, runs inference, and exposes decoded outputs.
* Deliver a collection of DSP primitives (windowing, spectral transforms, statistical features) which are commonly required by ML solutions on sensor data.
* Offer sample applications and CI-friendly examples that show how to package generated model sources, configure Kconfig options, and build with `west`.
* Simplify integration with external platforms and toolchains (model generation workflows, Edge AI Lab tooling, wrappers for third-party ML toolchains).
* Keep the runtime deterministic and suitable for resource-constrained devices (no dynamic allocation, minimal stack usage).

See the subpages for usage guides, API documentation and sample code.

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Subpages:

   applications/index
   samples/index
   integrations/index
   libraries/index
