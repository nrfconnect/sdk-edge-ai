.. _lib_model_ota:

Model-only OTA update library
##############################

.. contents::
   :local:
   :depth: 2

``lib/model_ota`` is a proof-of-concept library for updating *only* a Neuton or Axon NN model on a device that does not use mcuboot, without rebuilding or reflashing the rest of the application.

This document is the canonical explanation of how the library works, including step-by-step instructions for testing an OTA update for each backend (see "Testing an OTA update end-to-end" below).
For sample-specific details (Kconfig options, board overlays, expected sample output), see:

* :file:`samples/axon/hello_axon/README.rst` ("Model-only OTA update PoC") - Axon, minimal sample.
* :file:`samples/nrf_edgeai/regression/README.rst` ("Model-only OTA update") - both backends, folded into a real inference sample.

Overview
********

Normally, an nRF Edge AI model (Neuton or Axon) is compiled into the application image as C arrays: changing the model means rebuilding and reflashing the whole firmware.
This library lets an application instead load its model at runtime from a "model package" - a small header plus the model's payload - written to a dedicated ``model_storage`` flash partition, independently of the application image.

There is no runtime OTA transport in this PoC: getting a package onto the device is entirely a flash-only operation (for example, over SWD with ``nrfutil``), decoupled from the application.
The library itself is only concerned with what happens once bytes already sit in ``model_storage``:

* Validating the package (magic, container format version, model type, section sizes, CRC32) before trusting any of it.
* Wiring the payload into a working model instance the rest of nRF Edge AI can run inference against.
* Degrading gracefully - skipping inference and retrying periodically - if ``model_storage`` does not currently hold a valid package, rather than falling back to any compiled-in model.

The ``model_storage`` partition itself is not defined by this library: each sample or application provides its own devicetree overlay for it (typically repurposing an mcuboot-less board's unused second application slot).

Both samples that use this library build model-OTA in by default, but can opt out of it via a per-sample Kconfig option (:kconfig:option:`CONFIG_HELLO_AXON_MODEL_OTA` / :kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA`, both default ``y``) to restore their original, pre-model-OTA behavior instead: the model compiled directly into the application image, with no ``model_storage`` partition and no runtime loading.
See each sample's README.rst ("Making model OTA optional") for the exact build invocation.

The ``model_storage`` partition
=================================

None of the samples that use this library run mcuboot, so a board's second application slot (``slot1_partition``, normally reserved for mcuboot's image-swap update mechanism) sits entirely unused.
Every sample overlay deletes that node and repurposes the same flash region as a dedicated ``model_storage`` partition instead, sized generously for future (larger) models rather than reusing the much smaller default ``storage`` partition.
For example, see :file:`samples/axon/hello_axon/boards/nrf54lm20dk_nrf54lm20b_cpuapp.overlay`:

.. code-block:: dts

   /*
    * This sample does not use mcuboot, so the board's second application slot
    * (slot1_partition) is unused. Repurpose it as a dedicated "model_storage" partition.
    */
   /delete-node/ &slot1_partition;

   &cpuapp_rram {
       partitions {
           model_partition: partition@102000 {
               compatible = "zephyr,mapped-partition";
               label = "model_storage";
               reg = <0x102000 DT_SIZE_K(968)>;
           };
       };
   };

Since every loader and packaging tool addresses this partition by its devicetree node label (``model_partition``), a board that is missing this overlay, or that spells the label differently, would otherwise only surface as a much less obvious compile error deep inside a :file:`flash_map.h` macro expansion (or, worse, a working build that silently addresses the wrong flash region).
Both :file:`samples/axon/hello_axon/src/main.c` and :file:`samples/nrf_edgeai/regression/src/main.c` guard against this with a build-time assertion right where the partition is first used:

.. code-block:: c

   BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_partition),
                "board devicetree is missing the model_partition node - see boards/*.overlay");

Any new sample or application that adopts this library should add the same overlay pattern and ``BUILD_ASSERT`` for its own boards.

.. uml::
   :caption: Components and data flow. Host-side packaging is offline and backend-specific; on-device, only one backend's loader is actually built into a given image.

   skinparam shadowing false
   skinparam roundcorner 0
   skinparam backgroundColor #FFFFFF
   skinparam defaultTextAlignment center
   skinparam ArrowColor #0077C8
   skinparam ArrowThickness 1
   skinparam linetype ortho
   skinparam componentStyle rectangle

   skinparam component {
     BackgroundColor #13B6FF
     BorderColor #13B6FF
     FontColor #333F48
   }

   skinparam database {
     BackgroundColor #C1E8FF
     BorderColor #2149C2
     FontColor #333F48
   }

   left to right direction

   package "Host (build machine)" {
     component "Neuton model JSON\n(models/*.json)" as NeutonJson
     component "Reference build\nzephyr.elf" as RefElf
     component "package_model.py" as PkgNeuton
     component "package_model_axon.py" as PkgAxon
     component "model_v*.bin / .hex" as PkgFile
   }

   component "nrfutil device program" as Nrfutil

   package "Device" {
     database "model_storage\n(flash partition)" as ModelStorage

     package "lib/model_ota" {
       component "model_pkg_load_neuton()" as LoadNeuton
       component "model_pkg_load_axon()" as LoadAxon
     }

     component "nrf_edgeai_t\n(model.type = NEUTON | AXON)" as EdgeAi
     component "Application\n(inference loop, model_ota_load())" as App
   }

   NeutonJson --> PkgNeuton
   RefElf --> PkgAxon
   PkgNeuton --> PkgFile
   PkgAxon --> PkgFile
   PkgFile --> Nrfutil : flash independently\nof the application image
   Nrfutil --> ModelStorage

   ModelStorage --> LoadNeuton : XIP read + validate
   ModelStorage --> LoadAxon : XIP read + validate
   LoadNeuton --> EdgeAi : wire nrf_edgeai_model_neuton_t
   LoadAxon --> EdgeAi : wire nrf_axon_nn_compiled_model_s
   EdgeAi --> App

   legend bottom
     Only one backend's loader is actually built into a given device
     (CONFIG_MODEL_OTA_NEUTON xor CONFIG_MODEL_OTA_AXON, selected by the
     application); the two never compete for model_storage on the same
     device, only in this shared diagram.
   endlegend

Package format
***************

Both backends share the same *kind* of on-flash package - a fixed-size header (magic, format version, model type, name, model version, per-section byte lengths, CRC32) immediately followed by the payload sections it describes - but the two headers and payload layouts are otherwise independent and not interchangeable.
See :file:`include/model_ota/model_pkg.h` for the exact struct/enum definitions (``model_pkg_header`` / ``model_pkg_axon_header``, ``model_pkg_neuton_section`` / ``model_pkg_axon_section``).

Neuton packages
================

A Neuton package's payload is just the model's own raw arrays (weights, activation weights, output scaling, neuron/link topology, activation type mask), concatenated back-to-back in a fixed order, with no embedded absolute addresses anywhere.
Because nothing in the payload needs to know its own flash address, the on-device loader (``model_pkg_load_neuton()`` in :file:`model_pkg_neuton.c`) simply wires each section's flash address (found by offsetting from the start of the payload) directly into a ``nrf_edgeai_model_neuton_t``, plus a caller-provided RAM scratch buffer for neuron activations sized by :kconfig:option:`CONFIG_MODEL_OTA_MAX_NEURONS`.

.. uml::
   :caption: Neuton packaging and loading. No addresses are embedded in the payload, so packaging needs no reference build or ELF introspection.

   skinparam shadowing false
   skinparam roundcorner 0
   skinparam backgroundColor #FFFFFF
   skinparam ArrowColor #0077C8
   skinparam sequenceArrowThickness 1

   skinparam sequence {
     DividerBackgroundColor #8DBEFF
     DividerBorderColor #8DBEFF
     LifeLineBackgroundColor #13B6FF
     LifeLineBorderColor #13B6FF
     ParticipantBackgroundColor #13B6FF
     ParticipantBorderColor #13B6FF
     BoxBackgroundColor #C1E8FF
     BoxBorderColor #C1E8FF
     GroupBackgroundColor #8DBEFF
     GroupBorderColor #8DBEFF
   }

   skinparam note {
     BackgroundColor #ABCFFF
     BorderColor #2149C2
     Shadowing false
   }

   participant "Host:\npackage_model.py" as Host
   participant "model_storage\n(flash)" as Flash
   participant "model_pkg_load_neuton()" as Loader
   participant "Application" as App

   == Packaging (host, offline) ==

   Host -> Host : read model JSON\n(weights, topology, output scaling)
   Host -> Host : concatenate raw arrays\n(no addresses embedded)
   Host -> Host : build header\n(magic, format_version=3, section_len[], CRC32)
   note right of Host
     Neuton payload never embeds a flash
     address, so no reference build or
     ELF introspection is needed.
   end note

   == Flashing ==

   Host -> Flash : nrfutil device program model_v1.hex\n(independent of the application image)

   == Loading (device, at boot and periodically) ==

   App -> Loader : model_ota_load()
   Loader -> Flash : flash_area_read(header)
   Loader -> Loader : validate magic, format_version,\nmodel_type, section_len[] sum, CRC32
   alt package invalid or absent
     Loader --> App : NULL
     App -> App : skip inference,\nretry on next iteration
   else package valid
     Loader -> Flash : compute section pointers\n(offsets into payload)
     Loader -> Loader : wire nrf_edgeai_model_neuton_t\ndirectly at those flash addresses (XIP)
     Loader --> App : nrf_edgeai_t *
     App -> App : nrf_edgeai_run_inference()
   end

Axon packages
==============

An Axon package's payload is the model's *entire* compiler-generated ``nrf_axon_nn_compiled_model_s`` struct, captured byte-for-byte from a reference build, plus its command buffer and constants blob (``model_const``), and optionally an ``extra_outputs`` array.

Unlike Neuton, Axon's compiled-model struct is full of absolute pointers: the command buffer's own internal pointers into ``model_const``, and pointer fields inside the struct itself (``cmd_buffer_ptr``, ``model_const_ptr``, ``extra_outputs``, ``inputs[i].ptr``, ``output_ptr``).
Those pointers are only correct for the *reference build* they were captured from - they need to be rewritten (relocated) to wherever the corresponding data actually ends up once packaged.

This library resolves that relocation **once, on the host, at packaging time** (:file:`tools/model_ota/package_model_axon.py`), not on-device at every load:

* Pointers that refer to *flash-owned model data* (``cmd_buffer_ptr``, ``model_const_ptr``, ``extra_outputs``, and the command buffer's own internal pointers into ``model_const``) are rewritten to their final ``model_storage`` address directly, since that address is fixed and known ahead of time.
* Pointers that instead refer to *this device's own RAM* (``nrf_axon_interlayer_buffer``, for ``inputs[i].ptr`` / ``output_ptr``) cannot be resolved on the host, since that buffer's actual address is only known to the deployed firmware.
  These are reduced to a byte offset from the interlayer buffer's base instead, which the on-device loader adds back in.

Because the packaging tool captures and relocates the *entire* struct rather than a hand-picked subset of scalar fields, the on-device loader (``model_pkg_load_axon()`` in :file:`model_pkg_axon.c`) does not need to know a model's shape (input count, output count, ...) ahead of time - see "Known limitations" below for the shapes it still cannot handle.

.. uml::
   :caption: Axon packaging and loading. Flash-owned pointers are relocated once on the host; only RAM-owned pointers are patched on-device, and only cmd_buffer/model_const stay XIP.

   skinparam shadowing false
   skinparam roundcorner 0
   skinparam backgroundColor #FFFFFF
   skinparam ArrowColor #0077C8
   skinparam sequenceArrowThickness 1

   skinparam sequence {
     DividerBackgroundColor #8DBEFF
     DividerBorderColor #8DBEFF
     LifeLineBackgroundColor #13B6FF
     LifeLineBorderColor #13B6FF
     ParticipantBackgroundColor #13B6FF
     ParticipantBorderColor #13B6FF
     BoxBackgroundColor #C1E8FF
     BoxBorderColor #C1E8FF
     GroupBackgroundColor #8DBEFF
     GroupBorderColor #8DBEFF
   }

   skinparam note {
     BackgroundColor #ABCFFF
     BorderColor #2149C2
     Shadowing false
   }

   participant "Host:\nreference build\n(zephyr.elf)" as Ref
   participant "Host:\npackage_model_axon.py" as Host
   participant "model_storage\n(flash)" as Flash
   participant "model_pkg_load_axon()" as Loader
   participant "Application" as App

   == Packaging (host, offline) ==

   Ref -> Ref : CONFIG_..._REFERENCE_BUILD=y\nkeeps model_<name>, cmd_buffer_<name>,\naxon_model_const_<name> in the ELF
   Host -> Ref : pyelftools: read symbol bytes\n+ their original link addresses
   Host -> Host : memcpy the entire\nnrf_axon_nn_compiled_model_s (verbatim)
   Host -> Host : classify every pointer field:\nflash-owned (cmd_buffer, model_const, inputs[])\nvs RAM-owned (nrf_axon_interlayer_buffer,\noutput_ptr, persistent_vars)
   Host -> Host : relocate flash-owned pointers\nto their model_storage address\n(host-side, incl. cmd_buffer internal pointers)
   Host -> Host : rewrite RAM-owned pointers as\noffsets from nrf_axon_interlayer_buffer
   note right of Host
     Only offsets - not real addresses - are
     stored for RAM-owned fields, since the
     RAM buffer address is fixed at build time
     but the struct is packaged once and may be
     flashed onto builds with different addresses.
   end note
   Host -> Host : build header (magic, format_version=3,\nstruct_size, package_base, CRC32)

   == Flashing ==

   Host -> Flash : nrfutil device program model_v1.hex\n(independent of the application image)

   == Loading (device, at boot and periodically) ==

   App -> Loader : model_ota_load()
   Loader -> Flash : flash_area_read(header)
   Loader -> Loader : validate magic, format_version, struct_size,\npackage_base == partition address, CRC32
   alt package invalid or absent
     Loader --> App : NULL
     App -> App : skip inference,\nretry on next iteration
   else package valid
     Loader -> Flash : memcpy(out_model, MODEL_STRUCT, struct_size)\n(cmd_buffer / model_const stay XIP in flash)
     Loader -> Loader : reject unsupported shapes\n(labels != NULL, persistent_vars.count > 0)
     Loader -> Loader : patch RAM-owned pointers:\nadd nrf_axon_interlayer_buffer base\nto each stored offset
     Loader -> Loader : sanity-check cmd_buffer_ptr / model_const_ptr\nagainst computed section addresses
     Loader --> App : nrf_axon_nn_compiled_model_s *
     App -> App : nrf_axon_nn_model_validate() +\nnrf_axon_nn_run_inference()
   end

Host-side packaging tools
***************************

Model packages are produced by Python tools under :file:`tools/model_ota/`, run on the host, never on-device:

* :file:`package_model.py` - builds a Neuton package from a plain-JSON model description (see :file:`tools/model_ota/models/*.json` for examples).
  No ELF or reference build is needed, since a Neuton payload embeds no addresses.
  Useful for hand-edited/synthetic variants that have no corresponding generated source, such as :file:`tools/model_ota/models/regression_v2.json` (a deliberately hand-tweaked "retrained" variant).
* :file:`package_model_neuton.py` - builds a Neuton package directly from a Neuton codegen's generated model source (e.g. :file:`nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c` under any Neuton sample), regex-parsing its ``#define`` counts and ``static const`` coefficient arrays straight into a package - no hand-transcription into JSON, and (like :file:`package_model.py`) no ELF or reference build needed.
  This is the preferred way to package a freshly (re)trained Neuton model.
  Only ``MODEL_PARAMS_TYPE == f32`` models are supported (q16/q8 quantized models are rejected), and only regression/anomaly-detection tasks (only those generate the ``MODEL_OUTPUT_SCALE_MIN``/``MAX`` arrays the package format requires; classification-only models are rejected) - both limitations inherited from the package format itself, not specific to this tool.
* :file:`package_model_axon.py` - builds an Axon package purely by introspecting a *reference build's* ``zephyr.elf`` with ``pyelftools``: it locates the compiled model's symbols, reads the ``nrf_axon_nn_compiled_model_s`` struct's raw bytes with :file:`axon_struct_layout.py` (a ``ctypes`` mirror of the struct's on-device ABI), classifies every pointer field as flash-owned or RAM-owned, and relocates each accordingly (see "Axon packages" above).
  Every symbol lookup and cross-check (missing symbol, size/address mismatch against the struct's own fields, ...) fails loudly with a specific error at packaging time, on the host - so a stale or mismatched reference build is caught here, well before a package is ever written or flashed, with no separate build-time validation step needed.
* :file:`model_partition_layout.py` - shared helper both tools use to read the target ``model_storage`` partition's address and size from a build's generated ``zephyr.dts`` (via ``--dts``), and to preflight-check that the generated package will actually fit before writing anything - instead of trusting a hand-typed ``--address``/``--partition-size`` to still match the target board's devicetree.
  It also prints a one-line utilization report (used size, partition size, percentage used) after every successful build, e.g.:

  .. code-block:: console

     Partition region      Used Size  Region Size  %age Used
             model_storage:      48 KB      968 KB     4.96%

  so shrinking headroom is visible on every packaging run, well before a model actually stops fitting.
* :file:`test_package_model_axon.py` - host-side unit tests for the Axon tool's pointer classification/relocation logic, using synthetic model shapes (multi-input, ``extra_outputs``, rejected shapes) that do not exist as real models anywhere in this repo yet.
* :file:`test_package_model_neuton.py` - host-side unit tests for the Neuton source parser, checking it reproduces :file:`tools/model_ota/models/regression_v1.json` byte-for-byte from :file:`tools/model_ota/models/regression_v1_generated.c` (a restored copy of the generated source that JSON was originally hand-transcribed from), plus its error paths (wrong precision, wrong task, missing arrays).

All three packaging tools produce a ``.bin`` (raw package) and a ``.hex`` (the same bytes, addressed at the ``model_storage`` partition offset) - the latter is what gets flashed, independently of the application image, for example with:

.. code-block:: console

   nrfutil device program --firmware model_v1.hex \
       --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

``reset=RESET_SYSTEM`` matters: SWD/J-Link programming halts the CPU while writing, and ``nrfutil``'s post-program reset action otherwise defaults to ``RESET_NONE``, which leaves the core halted afterwards instead of resuming - this is what makes a freshly flashed board look like it "froze" until you press the reset button.

On-device loading (XIP)
*************************

Both loaders always load packages directly from the memory-mapped ``model_storage`` partition (execute-in-place, XIP) - there is no configurable RAM-copy loading mode.

* For Neuton, each array's flash address is wired directly into the model instance; the only RAM involved is the caller-provided neuron activation scratch buffer.
* For Axon, ``cmd_buffer`` and ``model_const`` are referenced straight in flash, regardless of model size - no RAM copy of either, and no runtime pointer-patching loop over the command buffer.
  Only the fixed-size ``nrf_axon_nn_compiled_model_s`` struct itself is copied, into the caller-owned model instance, since a handful of its fields need patching with this device's own ``nrf_axon_interlayer_buffer`` address (see "Axon packages" above).
  This is only possible because every other pointer field was already relocated to its final flash address at packaging time; the loader's only extra checks are that the packaged struct's size matches this firmware's (``struct_size``), and that the package's expected base address actually matches where it landed on this device (``package_base``).

Testing an OTA update end-to-end
***********************************

The steps below use the :file:`samples/nrf_edgeai/regression` sample, since it supports both backends on the nRF54LM20 DK (Neuton on ``nrf54lm20dk/nrf54lm20a/cpuapp``, Axon on ``nrf54lm20dk/nrf54lm20b/cpuapp``) and validates predictions against 29 known test cases, making it easy to see a model update actually change what the device predicts.
:file:`samples/axon/hello_axon` follows the same steps for Axon only, substituting its own sample path, Kconfig option, and model name (see its README for the exact commands).

In both cases, the pattern is the same: build and flash the *application* once, then repeatedly build/package/flash *only the model* without ever touching the application image again.

Testing a Neuton update
=========================

#. Build and flash the application (Neuton is the default backend on boards other than nRF54LM20, or select it explicitly with ``-DCONFIG_NRF_EDGEAI_REGRESSION_MODEL_NEUTON=y``):

   .. code-block:: console

      west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp -d build samples/nrf_edgeai/regression
      west flash -d build

#. Reset the board and observe the log; with an unprovisioned ``model_storage`` it repeats:

   .. code-block:: console

      No valid model in model_storage - waiting for one to be flashed. Inference is skipped until then.

#. Package the model this sample's Neuton backend already ships coefficients for, reading the partition layout from the build you just produced.
   If you have a freshly (re)trained model's generated source (``nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c``), package it directly - no hand-transcription into JSON needed:

   .. code-block:: console

      python3 tools/model_ota/package_model_neuton.py \
        tools/model_ota/models/regression_v1_generated.c \
        --name aq_regression --version 1.0.0 -o model_v1 \
        --dts build/regression/zephyr/zephyr.dts

   (:file:`regression_v1_generated.c` is a restored copy of this sample's original generated model source, used here as a stand-in for a real training run's output; a hand-written :file:`tools/model_ota/models/regression_v1.json` with identical coefficients is also still supported, via :file:`package_model.py`, for cases with no generated source to package from.)

#. Flash the package - independently of the application:

   .. code-block:: console

      nrfutil device program --firmware model_v1.hex --core Application \
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

#. Observe the log switch to running the 29 validation test cases (``Air quality - Predicted: ..., Expected: ..., absolute error ...``) every 5 seconds.

#. Repackage from :file:`tools/model_ota/models/regression_v2.json` (a hand-tweaked variant) as ``model_v2`` and flash it the same way, *without rebuilding or reflashing the application*.
   Observe the predicted/error values change on the next periodic reload (within 5 seconds) or after a reset.

Testing an Axon update
========================

#. Build and flash the application (Axon is the default backend on ``nrf54lm20b``, or select it explicitly with ``-DCONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON=y``):

   .. code-block:: console

      west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build samples/nrf_edgeai/regression
      west flash -d build

#. Reset the board and confirm it logs the same "No valid model in model_storage" message as the Neuton case above - the deployed application never links in a compiled-in Axon model either.

#. Build a *reference* image (never flashed) purely so the packaging tool has a real link address for the generated Axon model's struct, command buffer, and constants blob:

   .. code-block:: console

      west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build_ref samples/nrf_edgeai/regression -- \
        -DCONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON=y -DCONFIG_NRF_EDGEAI_REGRESSION_REFERENCE_BUILD=y

#. Package the model directly from the reference build's ELF (no separate header file needed):

   .. code-block:: console

      python3 tools/model_ota/package_model_axon.py \
        --elf build_ref/regression/zephyr/zephyr.elf \
        --model-name axon_user_instance_36025 -o model_v1 --version 1.0.0 \
        --dts build_ref/regression/zephyr/zephyr.dts

#. Flash the package - independently of the application:

   .. code-block:: console

      nrfutil device program --firmware model_v1.hex --core Application \
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

#. Observe the log switch to the same 29-test-case validation output as the Neuton case.

#. To observe an update, hand-tweak a constant in the reference build's generated model header (for example a bias or weight in :file:`src/nrf_edgeai_generated/Axon/nrf_edgeai_user_model_axon.h`), rebuild the reference image, repackage it as ``model_v2``, and flash only that to ``model_storage`` - again without touching the application image.
   The predicted/error values change on the next periodic reload or after a reset.

Known limitations
*******************

* **Axon models using** ``labels`` **(per-class text labels) or** ``persistent_vars`` **(streaming/** ``VarHandle`` **models) are rejected outright** by the packaging tool: resolving an arbitrary string literal's address reliably from an ELF, and computing per-persistent-variable buffer offsets, were judged not worth the added complexity for a PoC with no such model to validate against yet.
  The on-device loader also rejects them defensively (``MODEL_PKG_ERR_UNSUPPORTED_SHAPE``), in case a package somehow bypassed the tool's check.
* **Multi-input Axon models and models with** ``extra_outputs`` **are supported by both the tool and the loader**, but - absent such a model anywhere in this repo - are only exercised by :file:`tools/model_ota/test_package_model_axon.py`'s unit tests, not on real hardware.
* **No runtime OTA transport.** Getting a package onto the device is a flash-only operation in this PoC; there is no over-the-air download/verify/apply flow.
* **Single model instance.** Only one model package can live in ``model_storage`` at a time; there is no A/B slot, rollback, or versioned history.

Kconfig options
*****************

See :file:`Kconfig` for the authoritative list.
In summary:

* :kconfig:option:`CONFIG_MODEL_OTA` - master enable; depends on ``FLASH``, ``FLASH_MAP``, and ``CRC``.
* :kconfig:option:`CONFIG_MODEL_OTA_NEUTON` - builds :file:`model_pkg_neuton.c`.
  Off by default; applications select it explicitly so that ones which only ever load Axon packages do not build in this loader too.
* :kconfig:option:`CONFIG_MODEL_OTA_AXON` - builds :file:`model_pkg_axon.c`.
* :kconfig:option:`CONFIG_MODEL_OTA_MAX_NEURONS` - sizes the caller-side RAM scratch buffer for neuron activations; must be large enough for every Neuton model variant ever provisioned on a given device.
* :kconfig:option:`CONFIG_MODEL_OTA_LOG_LEVEL` - log level for the ``model_pkg`` module(s).

Integration pattern
**********************

A consuming application typically:

#. Declares a static, uninitialized-until-loaded model instance (``nrf_edgeai_model_neuton_t`` or ``nrf_axon_nn_compiled_model_s``) and a ``nrf_edgeai_t`` wrapping it, instead of pointing at a compiled-in model.
#. Calls ``model_pkg_load_neuton()`` or ``model_pkg_load_axon()`` at boot, and again periodically (or on some other trigger), to (re)validate and wire up whatever is currently in ``model_storage``.
#. Skips inference (rather than crashing) when the load fails, and retries on the next iteration.

See ``model_ota_load()`` in :file:`samples/axon/hello_axon/src/main.c` and :file:`samples/nrf_edgeai/regression/src/model_wiring_neuton.c` / :file:`model_wiring_axon.c` for concrete examples of this pattern for each backend.
