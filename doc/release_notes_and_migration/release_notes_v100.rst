.. _edgeai_release_notes_addon_v100:

Release notes for Edge AI Add-On v1.0.0
#######################################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the latest official release.
For more information refer to the following section.

For the list of potential issues, see the :ref:`edge_ai_known_issues` page.

Changelog
*********

This is an initial release of the |EAI|.
This release is based on the |NCS| release v3.2.0.

* Added:

  * An initial implementation of the nRF Edge AI library, including runtime, DSP, and NN modules.
  * Support for Cortex-M4 and Cortex-M33 architectures with precompiled libraries.
  * Three reference samples:

    * :ref:`Classification <runtime_classification_sample>`
    * :ref:`Regression <runtime_regression_sample>`
    * :ref:`Anomaly detection <runtime_anomaly_sample>`

  * Comprehensive documentation, including :ref:`integration <integrations>` guide, :ref:`library API references<libraries>`, and :ref:`samples overview <samples_nrf_edge_ai>`.
  * Kconfig options for enabling and configuring the library.
  * Build system integration with Zephyr and |NCS|.
  * Support for nRF52 DK, nRF52840 DK, nRF5340 DK, nRF54L15 DK, nRF54H20 DK, and nRF54LM20 DK.
