.. _edgeai_release_notes_addon_v201:

Release notes for Edge AI Add-On v2.0.1
#######################################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the :ref:`previous release <edgeai_release_notes_addon_v200>`.

For the list of potential issues, see the :ref:`edge_ai_known_issues` page.

Changelog
*********

.. note::
  The current release of the Edge AI Add-On is `experimental <software maturity_>`_.

This release is based on the |NCS| release v3.3.0-preview2.

This is a patch release on top of v2.0.0 that updates the Axon NPU software to v1.2.0 to restore compatibility with the `Nordic Edge AI Lab platform`_ web tool.

* Updated:

  * Axon NPU software to v1.2.0 (was v1.0.1 in v2.0.0).
    See the :ref:`axon_npu_changelog` page for the full list of changes.
    The most relevant changes since v2.0.0 are:

    * Compiler releases v1.1.0 and v1.2.0.
    * Support for multiple outputs in a model.
    * Support for the ``RESIZE_NEAREST_NEIGHBOR`` CPU operator.
    * ``static_assert`` in compiled model header files to verify that the interlayer buffer is allocated enough space to accommodate the model.
    * Compatibility check so that models report a minimum supported Axon version, preventing models compiled with new features from being run on an older version of the driver that does not support these features.
    * Option to print a histogram of bit differences between Axon inference and TFLite inference.
    * TFLite v2.19 as the officially supported version of TFLite. Version 2.15 should still work.
    * Build support for `nRF54LM20B`_ (board name ``nrf54lm20dk/nrf54lm20b/cpuapp``), replacing ``nRF54lm20a``.

  * :ref:`Axon NPU integration guide <axon_driver>` with sections on Axon NPU power management and the system resources used by the driver (workqueue, interrupt, semaphore, mutex).
  * :ref:`Axon NPU TFLite compiler documentation <axon_npu_tflite_compiler>` with a procedure for verifying a model on an Axon NPU-enabled device using the :ref:`test_nn_inference` test.
  * :ref:`Supported operators <supported_operators>` page with the ``Resize Nearest Neighbor`` operator and updated model structure constraints (maximum of 1 external input and 20 outputs).

* Fixed:

  * Quantization multiplier misapplied in some cases when one input to an ``Add`` operation is packed and the other is unpacked.
    You must recompile models to apply this fix.
  * Fully connected layers now work correctly with TFLite 2.19 for input lengths up to 2048 and output lengths up to 1024.
  * ``Sigmoid`` and ``Tanh`` after fully connected layers now work correctly with TFLite 2.19.

Compatibility
*************

* Axon NPU models compiled with previous versions of the compiler are compatible with the v1.2.0 driver.
  Models that use the ``Add`` operator must be recompiled to apply the quantization multiplier fix.
* Models compiled with the v1.2.0 compiler are compatible with older driver versions only if they do not use multiple outputs or the ``RESIZE_NEAREST_NEIGHBOR`` operation.
