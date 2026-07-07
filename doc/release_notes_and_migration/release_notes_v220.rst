.. _edgeai_release_notes_addon_v220:

Release notes for Edge AI Add-On v2.2.0
#######################################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the latest official release.
For more information refer to the following section.

You can also view detailed changelog pages for:

* :ref:`Axon NPU <axon_npu_changelog>`
* :ref:`nRF Edge AI Lib <nrf_edgeai_changelog>`

For the list of potential issues, see the :ref:`edge_ai_known_issues` page.

Changelog
*********

This release is based on the |NCS| release v3.4.0.

* Added:

  * :ref:`nRF Edge AI Observability Library <nrf_edgeai_obsv_lib>` with built-in metrics for tracking model performance at runtime, together with a host script for decoding collected metrics.
  * :ref:`Data forwarder sample application <data_forwarder_sample>`.
  * :ref:`Data forwarder host tool <data_forwarder_host_tool>` for receiving, decoding, and visualizing sensor data from the data forwarder sample.
  * Optional model observability in the :ref:`WW KWS application <app_ww_kws>` through the :ref:`nRF Edge AI Observability Library <nrf_edgeai_obsv_lib>`, enabled with the ``CONFIG_MODELS_OBSERVABILITY`` Kconfig option.

* Updated:

  * In the :ref:`Hello Edge Impulse sample application <hello_ei_sample>`, renamed the Kconfig option from ``CONFIG_EDGE_IMPULSE_PATH`` to ``CONFIG_EDGE_IMPULSE_MODEL_PATH``.
  * The keyword spotting stage in the :ref:`WW KWS application <app_ww_kws>` now uses class labels provided by the generated model.
  * Bundled Axon models are recompiled with a newer version of the Axon NPU compiler.

* Deprecated the :ref:`Edge Impulse data forwarder sample application <ei_data_forwarder_sample>`.
  This sample will be removed in the next release.
  Use the :ref:`Data forwarder sample <data_forwarder_sample>` instead and enable the ``CONFIG_DATA_FWD_PROTO_ASCII_MODE`` Kconfig option for |EI| CLI compatibility.
