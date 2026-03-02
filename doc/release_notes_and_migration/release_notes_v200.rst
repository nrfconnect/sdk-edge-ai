.. _edgeai_release_notes_addon_v200:

Release notes for Edge AI Add-On v2.0.0
#######################################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the latest official release.
For more information refer to the following section.

For the list of potential issues, see the :ref:`edge_ai_known_issues` page.

Changelog
*********

.. note::
  The current release of the Edge AI Add-On is `experimental <software maturity_>`_.

This release is based on the |NCS| release v3.3.0-preview2.

* Added:

  * :ref:`Hello Axon sample application <sample_hello_axon>`, along with documentation, demonstrating how to run neural model inference on the Axon NPU using the Axon NPU driver.
  * :ref:`Hello Edge Impulse sample application <hello_ei_sample>` demonstrating neural network inference using an |EI| machine learning model on the CPU and Axon NPU..
  * :ref:`Data forwarder sample application <ei_data_forwarder_sample>` demonstrating how to forward sensor data to |EIS|.
  * :ref:`Documentation for the Edge Impulse integration <edge_impulse_integration>`, with instructions for preparing and deploying |EI| machine learning models and using them in |EAI| applications.
  * Edge Impulse SDK v1.88.1 integrated into the |EAI| west manifest.
  * :ref:`Documentation for setting up the environment <setting_up_environment>`, depending on |EAI| use case.
  * Release configurations for the :ref:`app_gesture_recognition` application.
  * :ref:`Wakeword and Keyword Spotting application <app_ww_kws>`, demonstrating how to use wakeword model from `Nordic Edge AI Lab`_.
  * :ref:`Test: NN Inference application<test_nn_inference>`, demonstrating how to run and validate a compiled neural network model on an Axon‑enabled target.
  * :ref:`Gesture Recognition application <app_gesture_recognition>`, demonstrating how to use an nRF Edge AI model to recognize hand gestures from motion sensor data and expose them as standard HID inputs over Bluetooth® Low Energy.
    The application supports two execution backends: Neuton and Axon NPU.
  * Axon v1.0.1 support:
    
    * :ref:`Axon NPU compiler toolchain <axon_npu_tflite_compiler>`, located in :file:`tools/axon/compiler/scripts`.
      See the :ref:`axon_npu_changelog` for details.
    * :ref:`Axon NPU driver<axon_driver>` and :ref:`library code <lib_axon>`, located in :file:`drivers/axon` and :file:`lib/axon`.
    * Axon NPU :ref:`Inference test application <test_nn_inference>`, located in :file:`tests/axon/inference`.
