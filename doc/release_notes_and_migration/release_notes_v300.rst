.. _edgeai_release_notes_addon_v300:

Release notes for Edge AI Add-On v3.0.0
#######################################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the latest official release.

You can also view detailed changelog pages for:

* :ref:`Axon NPU <axon_npu_changelog>`
* :ref:`nRF Edge AI Lib <nrf_edgeai_changelog>`

For the list of potential issues, see the :ref:`edge_ai_known_issues` page.

Changelog
*********

This release is based on the |NCS| release v3.4.1.

* Added:

  * nRF Edge AI Lib v3.0.0 with split DSP feature extraction and feature scaling pipeline stages, runtime state tracking, and the public :c:func:`nrf_edgeai_process_features` API.
    See :ref:`nRF Edge AI Lib changelog <nrf_edgeai_changelog>` for migration details.
  * Axon NPU compiler release v2.0.0 with support for unfused GRU and the SUB operator.
    See :ref:`Axon NPU changelog <axon_npu_changelog>` for details.
  * Known issues filtering on the :ref:`edge_ai_known_issues` page, allowing you to view issues for a specific release.

* Updated:

  * Axon NPU to v2.0.1.
    All bundled Axon models were recompiled for the updated model description structure.
  * nRF Edge AI Lib to v3.0.0.
    Solutions and applications built for the 2.x runtime must be re-exported with Nordic Edge AI Lab 3.0.0.
  * The :ref:`Data forwarder sample <data_forwarder_sample>` to use gated sensor sampling when connected over Bluetooth LE, reducing average current consumption when not connected.
  * The :ref:`Data forwarder sample <data_forwarder_sample>` to align Bluetooth LE advertising parameters at startup and after disconnection.
  * The :ref:`Data forwarder host tool <data_forwarder_host_tool>` to v0.1.1, including a fix for truncated Y-axis labels on plots.
  * The :ref:`Gesture Recognition application <app_gesture_recognition>` with fixes for Thingy:53 release builds and an increased system workqueue stack size.

* Removed the deprecated Edge Impulse data forwarder sample application.
  Use the :ref:`Data forwarder sample <data_forwarder_sample>` instead and enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option for |EI| CLI compatibility.

* Fixed:

  * NCSDK-40932: Device Firmware Update (DFU) failure on Thingy:53 in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
