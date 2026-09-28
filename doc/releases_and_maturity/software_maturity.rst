.. _edgeai_software_maturity:

Software maturity
#################

.. contents::
   :local:
   :depth: 2

The |EAI| supports its libraries, integrations, samples, and applications at different software maturity levels.
This page summarizes the maturity of the main deliverables in the current documentation set.

Maturity definitions follow the |NCS| categories documented in :ref:`nrf:software_maturity`:

* **Supported** -- Implemented, maintained, and suitable for product development.
* **Not supported (--)** -- Not implemented or not maintained for this add-on.
* **Experimental** -- Available for development and evaluation, but not recommended for production.
  Interfaces and behavior may change between releases.

Library and integration maturity
********************************

The table below applies to |EAI| v|release_version|, based on |NCS| v|ncs_version_number|.
Maturity can change between add-on releases.
See :ref:`edgeai_release_notes` for announcements.

.. list-table:: Core component maturity
   :header-rows: 1
   :widths: 30 55 15

   * - Component
     - Scope
     - Maturity
   * - |EAILib| (Neuton CPU path)
     - End-to-end pipeline for Nordic Edge AI Lab Neuton models on Cortex-M4/M33 targets.
     - Supported
   * - |EAILib| (Axon NPU path)
     - End-to-end pipeline for Nordic Edge AI Lab Axon models on Axon-enabled SoCs.
     - Supported
   * - Axon NPU driver
     - On-device inference and DSP intrinsic execution through the Axon driver API.
     - Supported
   * - Axon NPU TFLite compiler
     - Host-side compilation of TFLite models into Axon model artifacts.
     - Supported
   * - Axon DSP intrinsics
     - Direct use of hardware-accelerated DSP primitives without a compiled neural network model.
     - Experimental
   * - Axon compiler advanced operators
     - Operators marked experimental in :ref:`axon_npu_changelog`, including unfused LSTM support.
     - Experimental
   * - |EI| integration
     - Deployment of |EI| models as a Zephyr module in |NCS| applications.
     - Supported
   * - nRF Edge AI observability
     - Runtime metrics collection through :ref:`nrf_edgeai_obsv_lib` and optional Memfault transport.
     - Experimental
   * - Data forwarder sample and host tool
     - Sensor streaming for dataset collection and debugging.
     - Supported

Applications
************

Applications in the |EAI| repository demonstrate complete product-style workflows.
:ref:`demos` in separate repositories are not maintained for every add-on release.

.. list-table:: Application maturity
   :header-rows: 1
   :widths: 30 55 15

   * - Application
     - Description
     - Maturity
   * - :ref:`app_gesture_recognition`
     - Gesture recognition over Bluetooth LE HID using Neuton or Axon models.
     - Supported
   * - :ref:`app_ww_kws`
     - Wakeword and keyword spotting with optional observability.
     - Supported
   * - :ref:`app_person_detection`
     - Camera-based person detection using sCAMIF and an Axon model.
     - Experimental

Versioning and compatibility
****************************

Software maturity describes readiness for production use.
It is separate from the version and compatibility rules for Axon NPU software and the |EAILib|.

Those components use independent version streams inside each add-on release.
For versioning, release cadence, and backward or forward compatibility expectations, see :ref:`release_model`.
