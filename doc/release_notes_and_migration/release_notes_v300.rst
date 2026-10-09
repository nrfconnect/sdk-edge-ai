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
  * The :ref:`nRF Edge AI Observability Library <nrf_edgeai_obsv_lib>` metrics, APIs, and CBOR wire format.

    Built-in output metrics are consolidated into two descriptors, and the on-wire schema is bumped from ``format_version`` 2 to ``3``.
    On-wire metric ids are renumbered to a dense ``1..5`` set.
    Each metric may carry an optional ``"c"`` key with the configuration its counters were gathered with (for example streak tuning for class predictions distribution, or mel scaling bounds for mel energy descriptor).
    The mel energy descriptor metric payload version is ``2``; other built-in metrics stay at version ``1``.

    When migrating from v2.3.0 observability payloads, Kconfig, or application code, apply the following mapping:

    * **Model certainty descriptor** (``model_certainty_desc``, id ``1``) replaces the prediction switching rate (id ``4``), probability entropy distribution (id ``5``), and probability top-2 margin distribution (id ``6``) metrics.
      Enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC`` instead of ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE``, ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST``, and ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST``.
      Set ``CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM`` (range ``4..16``) instead of the removed per-metric bin-count options.
      Replace ``nrf_edgeai_obsv_metric_psr_create``, ``nrf_edgeai_obsv_metric_ped_create``, and ``nrf_edgeai_obsv_metric_pmd_create`` with :c:func:`nrf_edgeai_obsv_metric_mcd_create`.
    * **Class predictions distribution** (``class_pred_dist``, id ``2``) replaces the probability distribution (id ``3``) and class streak distribution (id ``9``) metrics.
      Enable ``CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST`` instead of ``CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION`` and ``CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_STREAK_DIST``.
      Rename ``CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM`` to ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM``, ``CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOP`` to ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOP_BIN``, and ``CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOLERANCE`` to ``CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOL``.
      Replace ``nrf_edgeai_obsv_metric_pd_create`` and ``nrf_edgeai_obsv_metric_csd_create`` with :c:func:`nrf_edgeai_obsv_metric_cpd_create`.
    * **Transition matrix** (id ``3``), **mel energy descriptor** (id ``4``), and **mel spectral descriptor** (id ``5``) keep the same roles; their on-wire ids change from ``2/7/8`` to ``3/4/5``.
      :c:func:`nrf_edgeai_obsv_metric_tm_create`, :c:func:`nrf_edgeai_obsv_metric_med_create`, and :c:func:`nrf_edgeai_obsv_metric_msd_create` remain the integration entry points.

    Update the :file:`scripts/decode_edgeai_obsv_cdr/decode_edgeai_obsv_cdr.py` host decoder and any custom analysis tooling to accept ``format_version`` 3 and the new metric ids.
    The script rejects ``format_version`` 2 payloads because the id table changed.

    Memfault upload behavior changed for incremental metrics: :c:func:`nrf_edgeai_obsv_memfault_collect` encodes and resets each context in one critical section, a staged CDR is not overwritten (the function returns ``-EBUSY`` until Memfault drains the previous payload), and auto-collect retries after drain when a collect was skipped.
    Size ``CONFIG_SYSTEM_WORKQUEUE_STACK_SIZE`` for encode when ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT`` runs from the system workqueue.

  * The :ref:`WW KWS application <app_ww_kws>` observability integration to use separate per-model contexts in :file:`src/ww/ww_obsv.c` and :file:`src/kws/kws_obsv.c`.
    Enable ``CONFIG_MODELS_OBSERVABILITY_WW`` and/or ``CONFIG_MODELS_OBSERVABILITY_KWS`` instead of setting ``CONFIG_MODELS_OBSERVABILITY`` directly; the latter is now an auto-selected umbrella for shared options such as Memfault Diagnostic Service.
    When both models are observed, set ``CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS`` to at least ``2``.
    The keyword spotting model feeds mel features to the input-feature metrics; the wakeword score is expanded to a synthetic two-class ``[1 - p, p]`` vector for probability metrics.
    Generated model label symbols are renamed from ``MODEL_LABEL_INDEX_*`` to ``MODEL_USER_LABEL_*`` (including ``MODEL_USER_LABEL_COUNT``).
    See :file:`applications/ww_kws/observability.conf` for the default metric set and sizing used by the bundled models.
  * The :ref:`Data forwarder sample <data_forwarder_sample>` to use gated sensor sampling when connected over Bluetooth LE, reducing average current consumption when not connected.

* Removed the deprecated Edge Impulse data forwarder sample application.
  Use the :ref:`Data forwarder sample <data_forwarder_sample>` instead and enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option for |EI| CLI compatibility.

* Fixed:

  * NCSDK-40932: Device Firmware Update failure on Thingy:53 in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
  * System workqueue stack overflow in the :ref:`Gesture Recognition application <app_gesture_recognition>` during paring with ``CONFIG_BLE_MITM_AUTH`` Kconfig option enabled.
  * The :ref:`Data forwarder host tool <data_forwarder_host_tool>` truncating Y-axis labels on plots of individual channels.
