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
  * The :ref:`nrf_edgeai_obsv_lib` metrics, APIs, and CBOR wire format.

    Selected built-in metrics are consolidated.
    Probability distribution and Class streak distribution were merged into :ref:`nrf_edgeai_obsv_metrics_built_in_class_pred`.
    Prediction switching rate, Probability entropy distribution and Probability top-2 margin distribution were replaced by :ref:`nrf_edgeai_obsv_metrics_built_in_certainty`.
    Their respective Kconfig options were replaced with new ones.
    On-wire metric ids were renumbered and ``format_version`` was bumped to 3.

    The :ref:`decoding script <nrf_edgeai_obsv_script>`` was aligned to changes in metrics and CBOR wire format.
    It now accepts only the latest version of wire format.
  * The Memfault CDR transport for :ref:`nrf_edgeai_obsv_lib` to send incremental observability metrics.
    The :c:func:`nrf_edgeai_obsv_memfault_collect` function encodes and resets each observability context in one critical section if there is no staged CDR payload yet.
  * The :ref:`app_ww_kws` application observability integration to use all build-in metrics.
    The ``CONFIG_MODELS_OBSERVABILITY`` Kconfig option was replaced with ``CONFIG_MODELS_OBSERVABILITY_WW`` and ``CONFIG_MODELS_OBSERVABILITY_KWS`` Kconfig options so observability can be enabled separately for both models.
    The wakeword detection model prediction is expanded into syntetic two-class ``[1 - p, p]`` vector for probability metrics.
  * The :ref:`Data forwarder sample <data_forwarder_sample>` to use gated sensor sampling when connected over Bluetooth LE, reducing average current consumption when not connected.

* Removed the deprecated Edge Impulse data forwarder sample application.
  Use the :ref:`Data forwarder sample <data_forwarder_sample>` instead and enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option for |EI| CLI compatibility.

* Fixed:

  * NCSDK-40932: Device Firmware Update failure on Thingy:53 in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
  * System workqueue stack overflow in the :ref:`Gesture Recognition application <app_gesture_recognition>` during paring with ``CONFIG_BLE_MITM_AUTH`` Kconfig option enabled.
  * The :ref:`Data forwarder host tool <data_forwarder_host_tool>` truncating Y-axis labels on plots of individual channels.
