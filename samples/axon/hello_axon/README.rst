.. _sample_hello_axon:

Hello Axon
##########

.. contents::
   :local:
   :depth: 2

The Hello Axon sample demonstrates how to run neural model inference on the Axon NPU using the Axon NPU driver directly.
Use this sample if you are working directly with the Axon NPU compiler.

It also serves as a proof-of-concept for updating only the NN model, not the whole application, on a device that does not use mcuboot.

Requirements
************

The sample supports the following development kits:

.. table-from-sample-yaml::

Overview
********

The regression model runs on the Axon NPU and supports both synchronous and asynchronous inference modes.

The model uses Zephyr's `TensorFlow Lite for Microcontrollers: Hello World`_ sample.
Its task is to replicate the sine function in the range from 0 to 2π.
The TensorFlow Lite file describing this model is processed by the Axon NPU Compiler to convert it into a format accepted by the Axon NPU.
The compilation output is saved in :file:`src/generated/nrf_axon_model_hello_axon_.h`.

Unlike a typical Axon sample, the compiled model above is *not* linked into the application image: the app image contains only inference plumbing (Axon driver glue, quantization, logging), and loads its model from a dedicated flash partition, ``model_storage``, at runtime instead - see `Model-only OTA update PoC`_ below.

Model-only OTA update PoC
**************************

Normally, an Axon model is compiled into the application image as C arrays, so changing the model requires rebuilding and reflashing the whole firmware.
This sample instead reads a "model package" (see :file:`include/model_ota/model_pkg.h` in the |EAI| tree) from ``model_storage`` and validates it (magic number, container format version, model type, section sizes, CRC32) before every inference pass.

The package's payload is the model's *entire* compiler-generated ``nrf_axon_nn_compiled_model_s`` struct, captured byte-for-byte from a "model stub" - a tiny link of just the generated model header, produced automatically as part of a normal build (see below) - plus its command buffer and constants blob (``model_const``). Every pointer field in that struct, including ones that refer to this application's own RAM (e.g. the Axon interlayer buffer), is already the final, correct address by construction of that link, so the on-device loader performs no relocation of any kind. See :ref:`lib_model_ota` for a full explanation of the package format and the build-time mechanism that produces it.

If ``model_storage`` does not currently hold a valid package, the sample logs that state and skips inference instead of crashing, and keeps retrying periodically.
This means a new model can be provisioned - or a corrupted one recovered from - without a power cycle, though a reset also works.

The ``model_storage`` partition is a dedicated fixed partition, added by this sample's devicetree overlay (see :file:`boards`) on top of the board's second (unused, since mcuboot is not part of this PoC) application slot. It is sized generously (968 kB on the nRF54LM20 DK).

Preparing and flashing a model package
========================================

A model package (``hello_axon_model_pkg.bin``/``.hex``) is built automatically as part of a normal application build, once :kconfig:option:`CONFIG_HELLO_AXON_MODEL_OTA` is enabled (the default) - see :ref:`lib_model_ota` ("Build-time model packaging") for how :file:`CMakeLists.txt`'s ``nrf_axon_model_stub()`` call does this. No separate build or manual packaging step is needed:

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20b/cpuapp -d build samples/axon/hello_axon

This produces ``build/hello_axon/hello_axon_model_pkg.bin`` (the raw package) and ``.hex`` (the same bytes, addressed at the ``model_storage`` partition offset).

Flash the package independently of the application image, for example with:

.. code-block:: console

   nrfutil device program --firmware build/hello_axon/hello_axon_model_pkg.hex \
       --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

Programming over SWD/J-Link halts the CPU while writing; without ``reset=RESET_SYSTEM`` (``nrfutil``'s post-program reset action otherwise defaults to ``RESET_NONE``), the core is left halted afterwards instead of resuming, so it looks like the board "freezes" until you press the reset button.

Model package loading strategy
================================

``lib/model_ota`` always loads Axon packages directly from the memory-mapped ``model_storage`` partition (XIP; no RAM copy of ``cmd_buffer``/``model_const``, regardless of model size). See :ref:`lib_model_ota` ("On-device loading (XIP)") for the details.

Making model OTA optional
============================

Model-only OTA is enabled by default (:kconfig:option:`CONFIG_HELLO_AXON_MODEL_OTA` defaults to ``y``).
Build with it disabled to restore this sample's original, pre-model-OTA behavior instead: the model is compiled directly into the application image (from :file:`src/generated/nrf_axon_model_hello_axon_.h`), no ``model_storage`` partition or flash package is involved, and inference runs once at boot rather than being reloaded periodically.

.. code-block:: console

   west build -b nrf54lm20dk/nrf54lm20b/cpuapp samples/axon/hello_axon \
       -- -DCONFIG_HELLO_AXON_MODEL_OTA=n

Configuration
*************

|config|

Configuration options
=====================

|sample_kconfig|

.. options-from-kconfig::
   :show-type:

Building and running
********************

.. |sample path| replace:: :file:`samples/axon/hello_axon`

.. include:: /includes/build_and_run.txt

Testing
=======

|test_sample|

1. |connect_kit|
#. |connect_terminal_kit|
#. Build and flash the application, then flash the model package it produced (see above) to ``model_storage``.
#. Reset the development kit and observe the logging output; it should show predictions matching a normal (non-OTA) build of this sample.
#. Hand-tweak a constant in :file:`src/generated/nrf_axon_model_hello_axon_.h` (for example ``axon_model_const_hello_axon.l02_biasp``), rebuild, and flash only the newly-produced model package to ``model_storage`` - without reflashing the application.
#. Observe the logging output change to reflect the updated model, either after the next periodic reload or after a reset.

Sample output
-------------

The following output is logged in the terminal once a model package has been flashed:

   .. code-block:: console

      I: Hello Axon sample
      I: Initializing Axon NPU
      I: Loaded Axon model 'hello_axon' v0x00010000 (69 cmd words, 420 B const)
      I: Active model: 'hello_axon' version 0x00010000 (69 cmd words, 420 B const)
      I: Running asynchronous inference
      I: prediction:  0.051,  ideal  0.072
      I: prediction:  0.847,  ideal  0.842
      I: prediction: -0.491,  ideal -0.500
      ...

If ``model_storage`` does not hold a valid package, this is logged instead:

   .. code-block:: console

      W: No valid model in model_storage - waiting for one to be flashed. Inference is skipped until then.

Dependencies
************

This sample uses the following |EAI| libraries:

* :ref:`Axon NPU driver <axon_driver>`
* Model-only OTA update PoC library (:file:`lib/model_ota`, see :ref:`lib_model_ota`)

It uses the following Zephyr libraries:

* `Logging`_
* :file:`zephyr/storage/flash_map.h`
* :file:`zephyr/sys/crc.h`
