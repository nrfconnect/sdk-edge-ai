.. _edgeai_release_notes_addon_latest:

Release notes for Edge AI Add-On (latest)
#########################################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the latest official release.
For more information refer to the following section.

For the list of potential issues, see the :ref:`edge_ai_known_issues` page.

Changelog
*********

This release is based on the |NCS| release v3.2.0.

* Added:

  * :ref:`Hello Axon sample application <sample_hello_axon>`, along with documentation, demonstrating how to run neural model inference on the Axon NPU using the Axon NPU driver.
  * :ref:`Hello Edge Impulse sample application <hello_ei_sample>` demonstrating neural network inference using an |EI| machine learning model on the CPU.
  * :ref:`Data forwarder sample application <ei_data_forwarder_sample>` demonstrating how to forward sensor data to |EIS|.
  * :ref:`Documentation for the Edge Impulse integration <edge_impulse_integration>`, with instructions for preparing and deploying |EI| machine learning models and using them in |EAI| applications.
  * Edge Impulse SDK v1.82.3 integrated into the |EAI| west manifest.
  * :ref:`Documentation for setting up the environment <setting_up_environment>`, depending on |EAI| use case.
  * Release configurations for the :ref:`app_gesture_recognition` application.
  * :ref:`Wakeword and Keyword Spotting application <app_ww_kws>` demonstrating use of wakeword model from `Nordic Edge AI Lab`_.
  * :ref:`Test: NN Inference <test_nn_inference>` application, demonstrating how to run and validate a compiled neural network model on an Axon‑enabled target.
  * :ref:`Gesture Recognition application <app_gesture_recognition>`, demonstrating how to use an nRF Edge AI model to recognize hand gestures from motion sensor data and expose them as standard HID inputs over Bluetooth® Low Energy.
    The application supports two execution backends: Neuton and Axon NPU.
