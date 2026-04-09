.. _axon_npu_changelog:

Axon NPU changelog
##################

.. contents::
   :local:
   :depth: 2

This page tracks changes and updates as compared to the latest official release.
For more information refer to the following section.

For the list of Edge AI Add-on changelog information, see :ref:`edgeai_release_notes`.

Changelog
*********

See the list of changes for a specific release of the Axon NPU software.

Release 1.2.0  16 Apr 2026
******************************

* Compiler release 1.2.0
* Manifest (west.yml) pulls v3.3.0-preview2 tag of sdk-nrf (SHA-1 ede152ec21).
* Tested and built with SDK toolchain version v3.2.0. Users are recommended to use this version as well.

New features
------------
* Support for multiple outputs in a model.
* Support CPU operator RESIZE_NEAREST_NEIGHBOR.
* Compiled model header files will static_assert if the interlayer buffer is not allocated enough space to accomodate the model.
* Compatibility check: models will report a minimum supported axon version to prevent models compiled with new features from being run on an older version of the driver that does not support these features.
* Option to print histogram of bit differences between Axon inference and TFLite inference.

Bug fixes
------------
* Quantization multiplier misapplied under some circumstances when on input to Add operation is packed and the other is unpacked. Models must be recompiled to receive this fix.


Compatibility
-------------
* Older model versions are compatible with the latest driver. (Recommended that models using Add operator be recompiled due to bug mentioned above).
* New model versions are compatible with older driver versions if they do not include multiple outputs or RESIZE_NEAREST_NEIGHBOR operation.
  * Older driver versions will ignore all but the 1st output if a model has multiple outputs.
  * Older driver versions will not compile models with the RESIZE_NEAREST_NEIGBOR operation.

Release 1.1.0  19 Mar 2026
******************************

* Compiler release 1.1.0
* Manifest (west.yml) pulls v3.3.0-preview2 tag of sdk-nrf (SHA-1 ede152ec21).
* Tested and built with SDK toolchain version v3.2.0. Users are recommended to use this version as well.

New features
------------
* TFLite v2.19 is now the officially supported version of TFLite (2.15 should still work)
* Build support for nRF54lm20a is replaced with nRF54lm20b. New board name is 'nrf54lm20dk/nrf54lm20b/cpuapp' in build command.

Bug fixes
------------
* Fully connected layers with up to 2048 input length and 1024 outpu length fixed with TFLite 2.19 (previous version worked to 2046 and 512)
* Sigmoid and Tanh after fully connected  fixed for TFLite 2.19

Release 1.0.1  05 Mar 2026
==========================

* Added:

  * Compiler release 1.0.1

  .. note::
     The compiler was tested and built with SDK toolchain version v3.2.0.
     You should use this version for stable builds.

* Fixed:

  * Compiler now correctly handles a fully-connected layer followed by softmax when the fully-connected layer uses per-channel weight quantization (TensorFlow v2.19).

Known issues
------------

There are no reported known issues for this release.

Release 1.0.0  03 Mar 2026
==========================

This is the first public release of the Axon NPU software.

* Added:

  * Compiler release v1.0.0.
  * Support for v3.3.0-preview tag of the ``sdk-nrf`` (``SHA-1 ede152ec21``).

  .. note::
     The compiler was tested and built with SDK toolchain version v3.2.0.
     You should use this version for stable builds.

  * Arbitrary reshape operation.

* Updated:

  * Licenses to public Nordic license ``SPDX-License-Identifier: LicenseRef-Nordic-5-Clause``

* Fixed:

  * Softmax no longer specifies unpacked output.
    Affects a small number of models.

Known issues
------------

There are no reported known issues for this release.

Release 0.7.0  17 Feb 2026
==========================

* Added:

  * Compiler release v0.2.0.
  * Support for the ``main`` branch of ``sdk-nrf`` (tested with ``SHA-1 860c808db9``).

  .. note::
     The compiler was tested and built with SDK toolchain version v3.2.0.
     You should use this version for stable builds.

* Updated:

  * Modified the source tree to match ``sdk-edge-ai`` repository.
  * Improved performance in compiler and simulator when performing inference.

* Fixed:

  * Compiler now checks for dilation on depthwise convolution and returns errors.

Known issues
------------

Current release has the following known issues:

  * ADD operation with broadcast on both height and width axes is not supported when both inputs are dynamic (outputs from previous layers).
  * Per-layer testing in unit tests is not supported due to compilation errors caused by incompatible types.
