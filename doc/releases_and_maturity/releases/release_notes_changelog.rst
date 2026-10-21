.. _release_notes_addon_changelog:

Release notes for Edge AI Add-On main
#####################################

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

* Updated:

  * ``edge-impulse-sdk-zephyr`` in the west manifest to v1.95.28.
  * The :ref:`Hello Edge Impulse sample application <hello_ei_sample>` CPU model to version compatible with |EI| SDK.

* Removed:

  * The support for the ``nrf52dk/nrf52832``, ``nrf52840dk/nrf52840``, and ``nrf5340dk/nrf5340/cpuapp`` board targets in the following samples:

    * :ref:`runtime_classification_sample`
    * :ref:`runtime_anomaly_sample`
    * :ref:`runtime_regression_sample` -- removed ``nrf52dk/nrf52832``, ``nrf52840dk/nrf52840``, and ``nrf5340dk/nrf5340/cpuapp``.

  * The support for the ``thingy53/nrf5340/cpuapp`` board target in the :ref:`Gesture Recognition application <app_gesture_recognition>`.
