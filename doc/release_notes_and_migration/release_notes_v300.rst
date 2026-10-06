.. _release_notes_addon_v300:

Release notes for Edge AI Add-On v3.0.0
#######################################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the latest official release.

You can also view detailed changelog pages for:

* :ref:`Axon NPU <axon_npu_changelog>`
* :ref:`nRF Edge AI Lib <nrf_edgeai_changelog>`

For the list of potential issues, see the :ref:`known_issues` page.

Changelog
*********

This is the first release supported for production.
It is based on the |NCS| release v3.4.1.

* Updated:

  * Axon NPU to v2.0.1.
    All bundled Axon models were recompiled for the updated model description structure.
  * nRF Edge AI Lib to v3.0.0.
    Solutions and applications built for the 2.x runtime must be re-exported with Nordic Edge AI Lab 3.0.0.
  * The :ref:`Data forwarder sample <data_forwarder_sample>` to use gated sensor sampling when connected over Bluetooth LE, reducing average current consumption when not connected.

* Removed the deprecated Edge Impulse data forwarder sample application.
  Use the :ref:`Data forwarder sample <data_forwarder_sample>` instead and enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option for |EI| CLI compatibility.

* Fixed:

  * NCSDK-40932: Device Firmware Update failure on Thingy:53 in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
  * System workqueue stack overflow in the :ref:`Gesture Recognition application <app_gesture_recognition>` during paring with ``CONFIG_BLE_MITM_AUTH`` Kconfig option enabled.
  * The :ref:`Data forwarder host tool <data_forwarder_host_tool>` truncating Y-axis labels on plots of individual channels.
