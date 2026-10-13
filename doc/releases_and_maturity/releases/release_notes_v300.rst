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
  * The :ref:`nrf_edgeai_obsv_lib` metrics, APIs, and CBOR wire format:

    * Consolidated some of the built-in metrics:

      * Merged the probability distribution and class streak distribution metrics into :ref:`nrf_edgeai_obsv_metrics_built_in_class_pred`.
      * Replaced the prediction switching rate, probability entropy distribution, and probability top-2 margin distribution metrics with :ref:`nrf_edgeai_obsv_metrics_built_in_certainty`.

    * Replaced the Kconfig options of the affected metrics with new ones.
    * Renumbered the on-wire metric IDs and changed ``format_version`` to 3.

  * The :ref:`decoding script <nrf_edgeai_obsv_script>` to match the updated metrics and CBOR wire format.
    The script now accepts only the latest wire format version.
  * The Memfault CDR transport in :ref:`nrf_edgeai_obsv_lib`, which now sends incremental observability metrics.
    The :c:func:`nrf_edgeai_obsv_memfault_collect` function now encodes and resets each observability context in a single critical section, but only if no CDR payload is currently staged.
  * The :ref:`app_ww_kws` application observability integration to use all built-in metrics.

    * Replaced the ``CONFIG_MODELS_OBSERVABILITY`` Kconfig option with ``CONFIG_MODELS_OBSERVABILITY_WW`` and ``CONFIG_MODELS_OBSERVABILITY_KWS``.
      Observability can now be enabled separately for each model.
    * Expanded the wake word model prediction into a synthetic two-class ``[1 - p, p]`` vector for the probability metrics.

  * The :ref:`Data forwarder sample <data_forwarder_sample>` to use gated sensor sampling only when transport is enabled (Bluetooth LE connection is active and client is subscribed to notifications), reducing average current consumption when not connected.

* Removed the deprecated Edge Impulse data forwarder sample application.
  Use the :ref:`Data forwarder sample <data_forwarder_sample>` instead and enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option for |EI| CLI compatibility.

* Fixed:

  * NCSDK-40932: Device Firmware Update failure on Thingy:53 in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
    The root cause is NCSDK-21790 from the |NCS|.
  * NCSDK-40250: Restored serial recovery in the :ref:`Gesture Recognition application <app_gesture_recognition>` release configurations for Thingy:53.
  * System workqueue stack overflow in the :ref:`Gesture Recognition application <app_gesture_recognition>` during pairing with ``CONFIG_BLE_MITM_AUTH`` Kconfig option enabled.
  * The :ref:`Data forwarder host tool <data_forwarder_host_tool>` truncating Y-axis labels on plots of individual channels.
