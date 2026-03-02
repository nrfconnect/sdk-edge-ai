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
