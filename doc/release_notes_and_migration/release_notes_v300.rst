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
  * The :ref:`nRF Edge AI Observability Library <nrf_edgeai_obsv_lib>` metrics and CBOR wire format.

    Built-in metrics are consolidated into two descriptors, and the on-wire schema is bumped to ``format_version`` 3.
    Metric ids are renumbered to a dense ``1..5`` set, and each metric may carry an optional ``"c"`` key with the configuration its counters were gathered with.

    When migrating from v2.3.0 observability payloads or Kconfig, apply the following mapping:

    * **Model certainty descriptor** (``model_certainty_desc``, id ``1``) replaces the prediction switching rate, probability entropy distribution, and probability top-2 margin distribution metrics.
      Enable it with ``CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC`` instead of ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE``, ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST``, and ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST``.
      Set ``CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM`` instead of the removed per-metric bin-count options.
    * **Class predictions distribution** (``class_pred_dist``, id ``2``) replaces the probability distribution and class streak distribution metrics.
      Enable it with ``CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST`` instead of ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION`` and ``CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_STREAK_DIST``.
      Rename ``CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM`` to ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM``, ``CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOP`` to ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOP_BIN``, and ``CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOLERANCE`` to ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOL``.
    * **Transition matrix** (id ``3``), **mel energy descriptor** (id ``4``), and **mel spectral descriptor** (id ``5``) keep the same roles; only their on-wire ids change from ``2/7/8`` to ``3/4/5``.

    Update host-side decoders and analysis tooling to accept ``format_version`` 3 and the new metric ids.
    Payloads encoded with ``format_version`` 2 are not compatible with the v3.0.0 encoder layout.

  * The :ref:`WW KWS application <app_ww_kws>` observability integration to use separate per-model contexts and Kconfig options (``CONFIG_MODELS_OBSERVABILITY_WW`` and ``CONFIG_MODELS_OBSERVABILITY_KWS``) instead of the single ``CONFIG_MODELS_OBSERVABILITY`` switch.
    The keyword spotting model now feeds mel features to the input-feature metrics, and generated model label symbols are renamed from ``MODEL_LABEL_INDEX_*`` to ``MODEL_USER_LABEL_*``.
  * The :ref:`Data forwarder sample <data_forwarder_sample>` to use gated sensor sampling when connected over Bluetooth LE, reducing average current consumption when not connected.

* Removed the deprecated Edge Impulse data forwarder sample application.
  Use the :ref:`Data forwarder sample <data_forwarder_sample>` instead and enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option for |EI| CLI compatibility.

* Fixed:

  * NCSDK-40932: Device Firmware Update failure on Thingy:53 in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
  * System workqueue stack overflow in the :ref:`Gesture Recognition application <app_gesture_recognition>` during paring with ``CONFIG_BLE_MITM_AUTH`` Kconfig option enabled.
  * The :ref:`Data forwarder host tool <data_forwarder_host_tool>` truncating Y-axis labels on plots of individual channels.
