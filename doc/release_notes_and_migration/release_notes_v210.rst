.. _edgeai_release_notes_addon_v210:

Release notes for Edge AI Add-On v2.1.0
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

This release is based on the |NCS| release v3.3.0.

* Added:

  * :ref:`Axon Low Power sample application <sample_axon_low_power>` demonstrating energy efficiency of the Nordic Axon NPU.
  * :ref:`Person detection application <app_person_detection>` showing use of Axon NPU for real time person detection from a video stream.
  * Keyword spotting stage to the :ref:`WW KWS application <app_ww_kws>`.
  * Device firmware update (DFU) support in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
  * Axon model support in :ref:`Regression sample <runtime_regression_sample>`.

* Updated:

  * Axon NPU to v1.2.1 (see :ref:`Axon NPU changelog <axon_npu_changelog>` for details).
  * nRF Edge AI Lib to 2.2.1 (see :ref:`nRF Edge AI Lib changelog <nrf_edgeai_changelog>` for details).
  * Quantization function in the :ref:`Hello Axon sample application <sample_hello_axon>`.
  * The :ref:`Gesture Recognition application <app_gesture_recognition>` to use the `HID Service`_ from the |NCS| instead of a custom HID Service implementation.
  * Improved Bluetooth LE security of the :ref:`Gesture Recognition application <app_gesture_recognition>` with Bluetooth LE authentication.

* Removed the deprecated Partition Manager configuration from all applications, samples, and tests, except for the ``thingy53/nrf5340/cpuapp`` board target in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
