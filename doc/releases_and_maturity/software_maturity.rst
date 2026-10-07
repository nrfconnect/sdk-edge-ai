.. _software_maturity:

Software maturity
#################

.. contents::
   :local:
   :depth: 2


The tables below list maturity levels for |EAI| v|release_version|, based on |NCS| v|ncs_version_number|.
Maturity can change between add-on releases; see :ref:`release_notes` for announcements.

Maturity definitions follow the |NCS| categories documented in :ref:`nrf:software_maturity`:

* **Supported** -- Implemented, maintained, and suitable for product development.
* **Not supported (--)** -- Not implemented or not maintained for this add-on.
* **Experimental** -- Available for development and evaluation, but not recommended for production.
  Interfaces and behavior may change between releases.

Library and integration maturity
********************************

.. role:: subitem
   :class: sub-item

.. list-table:: Core component maturity
   :header-rows: 1
   :widths: 20 55 10

   * - Component
     - Scope
     - Maturity
   * - **Edge AI solution support**
     -
     -
   * - |EAILib|
     - End-to-end pipeline for Nordic Edge AI Lab models in |NCS| applications.
     - Supported
   * - :subitem:`with Axon`
     - Axon NPU path for models exported from Nordic Edge AI Lab.
     - Supported
   * - :subitem:`with Neuton`
     - Neuton CPU path for models on Cortex-M4/M33 targets.
     - Supported
   * - Axon NPU driver
     - On-device execution through the Axon driver API.
     - Supported
   * - :subitem:`Driver inference`
     - Inference on compiled neural network models.
     - Supported
   * - :subitem:`Driver intrinsics`
     - Discrete signal-processing algorithms using hardware-accelerated DSP primitives.
     - Supported
   * - Axon simulator
     - Host-based software simulator for Axon development and testing without hardware.
     - Supported
   * - |EI| integration
     - Deployment of |EI| models as a Zephyr module in |NCS| applications.
     - Experimental
   * - :subitem:`CPU`
     - |EI| model inference on Cortex-M CPU targets.
     - Experimental
   * - :subitem:`Axon`
     - |EI| model inference on Axon NPU targets.
     - Experimental
   * - **Edge AI feature support**
     -
     -
   * - Model observability
     - Runtime metrics collection through :ref:`nrf_edgeai_obsv_lib` and optional Memfault transport.
     - Experimental
   * - **Tool support**
     -
     -
   * - Axon NPU TFLite compiler
     - Host-side compilation of TFLite models into Axon model artifacts.
     - Supported
   * - :ref:`Data Forwarder Host tool <data_forwarder_host_tool>`
     - Desktop GUI for receiving, visualizing, and exporting sensor data from the data forwarder sample.
     - Experimental
