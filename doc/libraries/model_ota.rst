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
* :file:`applications/ww_kws/README.rst` - two independent Axon models on one device, each using CPU op extensions and ``persistent_vars`` - the real-world validation for those two features (see "Axon packages" below).

Overview
********

Normally, an nRF Edge AI model (Neuton or Axon) is compiled into the application image as C arrays: changing the model means rebuilding and reflashing the whole firmware.
This library lets an application instead load its model at runtime from a "model package" - a small header plus the model's payload - written to a dedicated ``model_storage`` flash partition, independently of the application image.

There is no runtime OTA transport in this PoC: getting a package onto the device is entirely a flash-only operation (for example, over SWD with ``nrfutil``), decoupled from the application.
The library itself is only concerned with what happens once bytes already sit in ``model_storage``:

* Validating the package (magic, container format version, model type, sizes, CRC32) before trusting any of it.
* Wiring the payload into a working model instance the rest of nRF Edge AI can run inference against.
* Failing loudly and skipping inference - rather than falling back to any compiled-in model - if ``model_storage`` does not currently hold a valid package.

The ``model_storage`` partition itself is not defined by this library: each sample or application provides its own devicetree overlay for it (typically repurposing an mcuboot-less board's unused second application slot), and passes its own partition (by devicetree node label) to ``model_pkg_load_neuton()`` / ``model_pkg_load_axon()``.
An application hosting more than one model (see :file:`applications/ww_kws`) simply defines one partition per model and loads each independently.

Every sample and application that uses this library builds model-OTA in by default, but can opt out of it via its own Kconfig option (:kconfig:option:`CONFIG_HELLO_AXON_MODEL_OTA` / :kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA` / :kconfig:option:`CONFIG_APP_MODEL_OTA`, all default ``y``) to restore its original, pre-model-OTA behavior instead: the model(s) compiled directly into the application image, with no ``model_storage`` partition(s) and no runtime loading.
See each sample's README.rst ("Making model OTA optional") for the exact build invocation.

The ``model_storage`` partition(s)
=====================================

None of the samples that use this library run mcuboot, so a board's second application slot (``slot1_partition``, normally reserved for mcuboot's image-swap update mechanism) sits entirely unused.
Every sample overlay deletes that node and repurposes the same flash region as one or more dedicated ``model_storage*`` partitions instead, sized generously for future (larger) models rather than reusing the much smaller default ``storage`` partition.
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

:file:`applications/ww_kws` follows the same pattern but splits the same region into two independent partitions, ``model_storage_ww`` and ``model_storage_kws``, one per model it hosts.

Since every loader and packaging tool addresses a partition by its devicetree node label, a board that is missing this overlay, or that spells the label differently, would otherwise only surface as a much less obvious compile error deep inside a :file:`flash_map.h` macro expansion (or, worse, a working build that silently addresses the wrong flash region).
Every sample/application that uses this library guards against this with a build-time assertion right where its partition(s) are first used, for example :file:`samples/axon/hello_axon/src/main.c`:

.. code-block:: c

   BUILD_ASSERT(FIXED_PARTITION_EXISTS(model_partition),
                "board devicetree is missing the model_partition node - see boards/*.overlay");

Any new sample or application that adopts this library should add the same overlay pattern and ``BUILD_ASSERT`` for its own boards.

.. uml::
   :caption: Components and data flow. Host-side packaging is offline and backend-specific; on-device, only one backend's loader is actually built into a given image (possibly loading more than one model of that backend, each from its own partition).

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
     component "Application's own\nzephyr.elf (no model linked in)" as AppElf
     component "Model stub\n(nrf_axon_model_stub())" as Stub
     component "package_model.py" as PkgNeuton
     component "package_model_axon.py" as PkgAxon
     component "model_v*.bin / .hex" as PkgFile
   }

   component "nrfutil device program" as Nrfutil

   package "Device" {
     database "model_storage*\n(flash partition(s))" as ModelStorage

     package "lib/model_ota" {
       component "model_pkg_load_neuton()" as LoadNeuton
       component "model_pkg_load_axon()" as LoadAxon
     }

     component "nrf_edgeai_t\n(model.type = NEUTON | AXON)" as EdgeAi
     component "Application\n(inference loop)" as App
   }

   NeutonJson --> PkgNeuton
   AppElf --> Stub : extract_elf_syms.py\n(app-owned symbol addresses)
   Stub --> PkgAxon
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
     device, only in this shared diagram. An application with several
     Axon models (e.g. ww_kws) builds one model stub, and calls
     model_pkg_load_axon() once, per model.
   endlegend

Package format
***************

Both backends share the same *kind* of on-flash package - a fixed-size header (magic, format version, model type, name, model version, size fields, CRC32) immediately followed by the payload it describes - but the two headers and payload layouts are otherwise independent and not interchangeable.
See :file:`include/model_ota/model_pkg.h` for the exact struct/enum definitions (``model_pkg_header`` / ``model_pkg_axon_header``, ``model_pkg_neuton_section``).

Neuton packages
================

A Neuton package's payload is just the model's own raw arrays (weights, activation weights, output scaling, neuron/link topology, activation type mask), concatenated back-to-back in a fixed order, with no embedded absolute addresses anywhere.
Because nothing in the payload needs to know its own flash address, the on-device loader (``model_pkg_load_neuton()`` in :file:`model_pkg_neuton.c`) simply wires each section's flash address (found by offsetting from the start of the payload) directly into a ``nrf_edgeai_model_neuton_t``, plus a caller-provided RAM scratch buffer for neuron activations sized by :kconfig:option:`CONFIG_MODEL_OTA_MAX_NEURONS`.

.. uml::
   :caption: Neuton packaging and loading. No addresses are embedded in the payload, so packaging needs no ELF introspection at all.

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
   Host -> Host : build header\n(magic, format_version, section_len[], CRC32)
   note right of Host
     Neuton payload never embeds a flash
     address, so no ELF introspection
     is needed.
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

An Axon package's payload is a "model stub"'s *entire linked memory image*, byte-for-byte - not a hand-picked, individually relocated subset of the compiler-generated ``nrf_axon_nn_compiled_model_s`` struct's fields.
This is what makes op extensions, ``persistent_vars``, and ``labels`` all "just work" without any special-casing in the packaging tool or the on-device loader: whatever the Axon NN compiler emitted for a given model is preserved and correctly addressed, whole.

The model stub, and why no relocation is needed
---------------------------------------------------

A **model stub** is a tiny, standalone link of nothing but a generated Axon model header (:file:`nrf_axon_model_<name>.h`), produced automatically as part of a normal ``west build`` by ``nrf_axon_model_stub()`` (:file:`lib/model_ota/cmake/nrf_axon_model_stub.cmake`) - no separate build step, and no "reference build" of the whole application.
Crucially, the *deployed application* never links the generated header in at all: it only ever calls ``model_pkg_load_axon()`` against whatever is currently in ``model_storage``.

A generated Axon model header references three kinds of symbols it does not itself define, because they are owned by whatever application eventually deploys the model:

* ``nrf_axon_interlayer_buffer`` - RAM scratch shared by every model on the device.
* ``nrf_axon_nn_op_extension_*`` - CPU fallback functions for NPU operations the model uses (for example ``softmax``, ``sigmoid_v2`` - see :file:`drivers/axon/nrf_axon_nn_op_extensions.c` for what these are).
* ``axonpro_*`` driver-owned lookup-table constants some ``cmd_buffer`` entries reference directly (for example ``axonpro_int8_packing_filter``) - fixed tables baked into the Axon driver library, not into the application, but still only resolvable once something has actually linked that driver in.
* ``axon_model_<name>_persistent_vars`` - the RAM backing store for a model's persistent variables (streaming/recurrent models), if it has any. Unlike the first three, the generated header *defines* this array itself (not ``extern``), since ordinarily the header is compiled straight into the application that owns it.

``nrf_axon_model_stub()`` builds the model stub in three steps:

#. **Discover** which of the above symbols a given model's header actually references (:file:`tools/model_ota/gen_axon_stub_fixups.py`), and force them to stay resolvable in the *deployed application's* link via ``toolchain_ld_force_undefined_symbols()`` - otherwise nothing in the application references them either, and the linker's ``--gc-sections`` would happily discard them.
#. Once the application's own ``zephyr.elf`` exists, **extract the real address** each of those symbols ended up at in *this specific build* (:file:`tools/model_ota/extract_elf_syms.py`, via ``nm``), and turn that into a ``PROVIDE()`` linker-script fragment.
#. **Compile and link** a patched copy of the model header (``persistent_vars`` array definitions rewritten to ``extern``, so the stub references the application's own copy instead of defining a second one) at the target ``model_storage*`` partition's own address, feeding it that ``PROVIDE()`` script.

Once linked this way, *every* pointer field the model header's data carries - flash-owned (``cmd_buffer``, ``model_const``, ``labels``, ``extra_outputs``) or app-owned RAM (``nrf_axon_interlayer_buffer``, ``persistent_vars``, op-extension function pointers embedded in ``cmd_buffer``) - is already the final, correct absolute address for this exact application, because the *compiler and linker* computed it, not host-side Python arithmetic.
There is nothing left to relocate.

:file:`tools/model_ota/package_model_axon.py` therefore does far less work than a from-scratch relocating packager would: it copies the model stub's ``.model_stub`` output section verbatim as the package payload, and just records where the ``nrf_axon_nn_compiled_model_s`` struct starts within it (``struct_offset``) plus the address the stub was linked at (``package_base``, for the on-device loader to cross-check against).

.. uml::
   :caption: Axon packaging and loading via the second-pass link. Every pointer is already correct by construction of the model stub's own link; nothing is relocated on the host or patched on-device.

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

   participant "Host:\napplication build\n(zephyr.elf, no model linked in)" as AppElf
   participant "Host:\nnrf_axon_model_stub()" as Stub
   participant "Host:\npackage_model_axon.py" as Host
   participant "model_storage*\n(flash)" as Flash
   participant "model_pkg_load_axon()" as Loader
   participant "Application" as App

   == Build (host, part of a normal west build) ==

   Stub -> Stub : gen_axon_stub_fixups.py:\ndiscover app-owned symbols,\npatch persistent_vars to extern
   AppElf -> Stub : extract_elf_syms.py:\nnm real addresses of those symbols\nin *this* application's own link
   Stub -> Stub : compile+link patched header\nat model_storage* partition address,\nPROVIDE()-ing those real addresses
   note right of Stub
     Every pointer field - flash-owned or
     app-owned RAM - is now the final,
     correct absolute address, computed
     by the compiler/linker, not Python.
   end note
   Stub -> Host : model stub ELF (.model_stub section)

   == Packaging (host, offline) ==

   Host -> Host : copy .model_stub section verbatim\n(no relocation needed)
   Host -> Host : record struct_offset, package_base,\nstruct_size, CRC32 in the header

   == Flashing ==

   Host -> Flash : nrfutil device program model_v1.hex\n(independent of the application image)

   == Loading (device, at boot) ==

   App -> Loader : model_pkg_load_axon(fa_id, partition_addr, ...)
   Loader -> Flash : flash_area_read(header)
   Loader -> Loader : validate magic, format_version, struct_size,\npackage_base == partition address, CRC32
   alt package invalid or absent
     Loader --> App : error
     App -> App : skip inference
   else package valid
     Loader -> Flash : memcpy(out_model, struct_offset, struct_size)\n(cmd_buffer / model_const stay XIP in flash)
     Loader -> Loader : sanity-check cmd_buffer_ptr / model_const_ptr\nfall within this package's payload
     Loader --> App : nrf_axon_nn_compiled_model_s *
     App -> App : nrf_axon_nn_model_validate() +\nnrf_axon_nn_run_inference()
   end

Because the model stub already resolved every pointer, ``model_pkg_load_axon()`` (:file:`model_pkg_axon.c`) does not need to know a model's shape (input count, output count, whether it uses ``persistent_vars`` or ``labels``, ...) ahead of time, and performs no relocation of any kind at load time - only the checks in the diagram above, all of them either integrity checks (CRC32) or defense-in-depth sanity checks against a package built for the wrong partition/firmware.

Host-side packaging tools
***************************

Model packages are produced by Python tools under :file:`tools/model_ota/`, run on the host, never on-device:

* :file:`package_model.py` - builds a Neuton package from a plain-JSON model description (see :file:`tools/model_ota/models/*.json` for examples).
  No ELF is needed, since a Neuton payload embeds no addresses.
  Useful for hand-edited/synthetic variants that have no corresponding generated source, such as :file:`tools/model_ota/models/regression_v2.json` (a deliberately hand-tweaked "retrained" variant).
* :file:`package_model_neuton.py` - builds a Neuton package directly from a Neuton codegen's generated model source (e.g. :file:`nrf_edgeai_generated/Neuton/nrf_edgeai_user_model.c` under any Neuton sample), regex-parsing its ``#define`` counts and ``static const`` coefficient arrays straight into a package - no hand-transcription into JSON, and (like :file:`package_model.py`) no ELF needed.
  This is the preferred way to package a freshly (re)trained Neuton model. Called automatically by ``nrf_neuton_model_package()`` (:file:`lib/model_ota/cmake/nrf_neuton_model_package.cmake`) as part of a normal ``west build`` for any sample that opts in - see :file:`samples/nrf_edgeai/regression/CMakeLists.txt` - reading the target partition's address/size straight from devicetree rather than needing a completed build's ``zephyr.dts`` via ``--dts`` (that option remains available for packaging outside of a CMake-integrated build, e.g. a different model than the one a sample currently bundles).
  Only ``MODEL_PARAMS_TYPE == f32`` models are supported (q16/q8 quantized models are rejected), and only regression/anomaly-detection tasks (only those generate the ``MODEL_OUTPUT_SCALE_MIN``/``MAX`` arrays the package format requires; classification-only models are rejected) - both limitations inherited from the package format itself, not specific to this tool.
* :file:`gen_axon_stub_fixups.py` - discovers the app-owned symbols (interlayer buffer, op extensions, ``axonpro_*`` constants, ``persistent_vars``) a given model header references, and produces the patched copy of that header the model stub actually compiles (see "Axon packages" above). Called automatically by ``nrf_axon_model_stub()``, both at CMake configure time (to force-keep the discovered symbols alive) and at build time (to regenerate the patched header if the model header itself changes).
* :file:`extract_elf_syms.py` - ``nm``'s a built ``zephyr.elf`` for a given symbol list's real addresses, and emits a ``PROVIDE()`` linker-script fragment (setting the Thumb bit for function symbols, i.e. op extensions). Called automatically by ``nrf_axon_model_stub()`` once the application's own ELF exists.
* :file:`package_model_axon.py` - builds an Axon package from a model stub's ELF (see "Axon packages" above): it locates the ``.model_stub`` output section and the ``nrf_axon_nn_compiled_model_s`` struct's raw bytes and address within it with :file:`axon_struct_layout.py` (a ``ctypes`` mirror of the struct's on-device ABI), validates the model's shape is supported (rejects a non-``NULL`` ``packed_output_buf`` - see "Known limitations" below), and writes the header plus payload verbatim - no relocation.
  Every check (missing symbol, size/address mismatch, package outside the target partition, ...) fails loudly with a specific error at packaging time, on the host - so a stale or mismatched build is caught here, well before a package is ever written or flashed.
* :file:`model_partition_layout.py` - shared helper both Axon and Neuton tools use to read the target ``model_storage`` partition's address and size from a build's generated ``zephyr.dts`` (via ``--dts``), and to preflight-check that the generated package will actually fit before writing anything - instead of trusting a hand-typed ``--address``/``--partition-size`` to still match the target board's devicetree.
  It also prints a one-line utilization report (used size, partition size, percentage used) after every successful build, e.g.:

  .. code-block:: console

     Partition region      Used Size  Region Size  %age Used
             model_storage:      48 KB      968 KB     4.96%

  so shrinking headroom is visible on every packaging run, well before a model actually stops fitting.
* :file:`test_package_model_axon.py` - host-side unit tests for the Axon tool's model-shape validation and on-flash header format, using synthetic ``nrf_axon_nn_compiled_model_s`` instances built directly from raw bytes (no ELF needed for these).
* :file:`test_gen_axon_stub_fixups.py` / :file:`test_extract_elf_syms.py` - host-side unit tests for the model-stub symbol-discovery/header-patching and ELF-symbol-extraction logic, using synthetic header text and a mocked ``nm`` output respectively.
* :file:`test_package_model_neuton.py` - host-side unit tests for the Neuton source parser, checking it reproduces :file:`tools/model_ota/models/regression_v1.json` byte-for-byte from :file:`tools/model_ota/models/regression_v1_generated.c` (a restored copy of the generated source that JSON was originally hand-transcribed from), plus its error paths (wrong precision, wrong task, missing arrays).

All packaging tools produce a ``.bin`` (raw package) and a ``.hex`` (the same bytes, addressed at the target partition offset) - the latter is what gets flashed, independently of the application image, for example with:

.. code-block:: console

   nrfutil device program --firmware model_v1.hex \
       --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

``reset=RESET_SYSTEM`` matters: SWD/J-Link programming halts the CPU while writing, and ``nrfutil``'s post-program reset action otherwise defaults to ``RESET_NONE``, which leaves the core halted afterwards instead of resuming - this is what makes a freshly flashed board look like it "froze" until you press the reset button.

For samples that opt in, packaging itself is not a manual step for either backend: ``nrf_axon_model_stub()`` and ``nrf_neuton_model_package()`` both run their respective packaging tool automatically as part of a normal ``west build`` (see "Axon packages" above and "Neuton/Axon alignment" below), so the ``.bin``/``.hex`` files land straight in the build directory (e.g. :file:`build/<sample>/<target>_model_pkg.hex`) - only the flashing step above is manual.

On-device loading (XIP)
*************************

Both loaders always load packages directly from the memory-mapped partition (execute-in-place, XIP) - there is no configurable RAM-copy loading mode.

* For Neuton, each array's flash address is wired directly into the model instance; the only RAM involved is the caller-provided neuron activation scratch buffer.
* For Axon, ``cmd_buffer`` and ``model_const`` are referenced straight in flash, regardless of model size - no RAM copy of either.
  Only the fixed-size ``nrf_axon_nn_compiled_model_s`` struct itself is copied, into the caller-owned model instance, since it is the one part of the payload the application needs as a normal (not memory-mapped) C struct.

``model_pkg_load_axon()`` takes the target partition as parameters (a flash-area ID and its memory-mapped base address - see :c:func:`PARTITION_ID` / :c:func:`PARTITION_ADDRESS`), rather than hard-coding a single ``model_storage`` label, precisely so that an application hosting several independent Axon models (see :file:`applications/ww_kws`) can call it once per model, against each one's own partition.
``model_pkg_load_neuton()`` does not: it still hard-codes the ``model_partition`` devicetree label internally, since no sample or application hosts more than one Neuton model yet - see "Neuton/Axon alignment" below.

Neuton/Axon alignment
************************

Both backends share the same header "shape" (magic, format version, model type, name, model version, size fields, CRC32) and the same fail-closed loading behavior, but are otherwise independent, and deliberately not aligned everywhere:

* **Build-time packaging is now symmetric.** ``nrf_neuton_model_package()`` (:file:`lib/model_ota/cmake/nrf_neuton_model_package.cmake`) mirrors ``nrf_axon_model_stub()``: both run automatically as part of a normal ``west build`` and drop a ready-to-flash ``.bin``/``.hex`` straight in the build directory, with no separate manual packaging step for either backend's samples.
  Neuton needs far less machinery to get there - no ELF, no model stub, no second-pass link - since a Neuton package never embeds an address in the first place (see "Neuton packages" above); it only needs the target partition's address/size, read from devicetree at configure time exactly like ``nrf_axon_model_stub()`` does.
* **The loader signature is deliberately not symmetric.** ``model_pkg_load_axon()`` takes its partition as parameters; ``model_pkg_load_neuton()`` still hard-codes the ``model_partition`` devicetree label (see "On-device loading (XIP)" above).
  Parameterizing it the same way would be a small change, but nothing in this repo needs a second Neuton model yet, unlike Axon where :file:`applications/ww_kws` concretely does; this is tracked here as a known, intentional asymmetry rather than implemented speculatively.

Testing an OTA update end-to-end
***********************************

The steps below use the :file:`samples/nrf_edgeai/regression` sample, since it supports both backends on the nRF54LM20 DK (Neuton on ``nrf54lm20dk/nrf54lm20a/cpuapp``, Axon on ``nrf54lm20dk/nrf54lm20b/cpuapp``) and validates predictions against 29 known test cases, making it easy to see a model update actually change what the device predicts.
:file:`samples/axon/hello_axon` follows the same steps for Axon only, substituting its own sample path, Kconfig option, and model name (see its README for the exact commands).
:file:`applications/ww_kws` (see its own testing steps below) additionally validates two independent models, each using CPU op extensions and ``persistent_vars``, on one device.

In all cases, the pattern is the same: build the *application* once (packaging now happens automatically as part of that same build, for both backends - see "Host-side packaging tools" above), flash it, then repeatedly rebuild/reflash *only the model package* without ever touching the application image again.

Testing a Neuton update
=========================

#. Build and flash the application (Neuton is the default backend on boards other than nRF54LM20, or select it explicitly with ``-DCONFIG_NRF_EDGEAI_REGRESSION_MODEL_NEUTON=y``).
   This single build already produces the model package too - watch the end of the build log for ``model_ota: packaging regression (aq_regression, Neuton)``:

   .. code-block:: console

      west build -p -b nrf54lm20dk/nrf54lm20a/cpuapp -d build samples/nrf_edgeai/regression
      west flash -d build

#. Reset the board and observe the log; with an unprovisioned ``model_storage`` it repeats:

   .. code-block:: console

      No valid model in model_storage - waiting for one to be flashed. Inference is skipped until then.

#. Flash the package the build already produced - independently of the application:

   .. code-block:: console

      nrfutil device program --firmware build/regression/regression_model_pkg.hex --core Application \
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

#. Observe the log switch to running the 29 validation test cases (``Air quality - Predicted: ..., Expected: ..., absolute error ...``) every 5 seconds.

#. To package a different model instead of this sample's bundled :file:`tools/model_ota/models/regression_v1_generated.c` (for example a freshly (re)trained one), run :file:`package_model_neuton.py` directly against its own generated source - no hand-transcription into JSON needed:

   .. code-block:: console

      python3 tools/model_ota/package_model_neuton.py \
        path/to/nrf_edgeai_user_model.c \
        --name aq_regression --version 1.0.0 -o model_v1 \
        --dts build/regression/zephyr/zephyr.dts

   and flash the resulting ``model_v1.hex`` the same way, again without touching the application image.

#. Repackage from :file:`tools/model_ota/models/regression_v2.json` (a hand-tweaked variant with no corresponding generated source) using :file:`package_model.py` instead:

   .. code-block:: console

      python3 tools/model_ota/package_model.py \
        tools/model_ota/models/regression_v2.json -o model_v2 \
        --dts build/regression/zephyr/zephyr.dts

   and flash it the same way, *without rebuilding or reflashing the application*.
   Observe the predicted/error values change on the next periodic reload (within 5 seconds) or after a reset.

Testing an Axon update
========================

#. Build and flash the application (Axon is the default backend on ``nrf54lm20b``, or select it explicitly with ``-DCONFIG_NRF_EDGEAI_REGRESSION_MODEL_AXON=y``).
   This single build already produces the model package too - watch the end of the build log for ``model_ota: packaging regression (axon_user_instance_36025)``:

   .. code-block:: console

      west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build samples/nrf_edgeai/regression
      west flash -d build

#. Reset the board and confirm it logs the same "No valid model in model_storage" message as the Neuton case above - the application never links in a compiled-in Axon model either.

#. Flash the package the build already produced - independently of the application:

   .. code-block:: console

      nrfutil device program --firmware build/regression/regression_model_pkg.hex --core Application \
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

#. Observe the log switch to the same 29-test-case validation output as the Neuton case.

#. To observe an update, hand-tweak a constant in the generated model header (for example a bias or weight in :file:`src/nrf_edgeai_generated/Axon/nrf_edgeai_user_model_axon.h`), rebuild (``west build``, no ``-p`` needed), and flash the newly produced package the same way, again without touching the application image.
   The predicted/error values change on the next periodic reload or after a reset.

Testing ww_kws (two models, op extensions, persistent_vars)
================================================================

:file:`applications/ww_kws` hosts two independent Axon models - wakeword detection and keyword spotting - each in its own ``model_storage_ww`` / ``model_storage_kws`` partition, and each exercising a feature the simpler samples above do not: the wakeword model uses a CPU op extension (``nrf_axon_nn_op_extension_sigmoid_v2``) and the keyword-spotting model uses both a different op extension (``nrf_axon_nn_op_extension_softmax``) and ``persistent_vars`` (both models actually use ``persistent_vars``, for their streaming audio-feature state).

#. Build and flash the application (:kconfig:option:`CONFIG_APP_MODEL_OTA` defaults to ``y``; this single build produces both model packages):

   .. code-block:: console

      west build -p -b nrf54lm20dk/nrf54lm20b/cpuapp -d build applications/ww_kws
      west flash -d build

#. Reset the board and observe both models fail to load (nothing is flashed to either partition yet):

   .. code-block:: console

      No usable WW model in model_storage_ww (err -3)
      No usable KWS model - see model_storage_kws flashing instructions in doc/libraries/model_ota.rst

#. Flash both packages the build already produced - independently of the application:

   .. code-block:: console

      nrfutil device program --firmware build/ww_kws/ww_model_pkg.hex --core Application \
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_NONE
      nrfutil device program --firmware build/ww_kws/kws_model_pkg.hex --core Application \
        --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM

#. Observe both models load successfully and the application complete initialization:

   .. code-block:: console

      Loaded Axon model 'axon_user_instan' v0x00010000 (2432 cmd words, 24996 B const)
      Active WW model: 'axon_user_instan' version 0x00010000 (2432 cmd words, 24996 B const)
      Loaded Axon model 'axon_user_instan' v0x00010000 (15726 cmd words, 295936 B const)
      Active KWS model: 'axon_user_instan' version 0x00010000 (15726 cmd words, 295936 B const)
      Initialization completed, check output on VCOM0

#. Continue with the application's own testing steps (see :file:`applications/ww_kws/README.rst`, "Testing") - say the wakeword phrase, then a keyword - to confirm both OTA-loaded models, including their op extensions and ``persistent_vars`` state, actually run inference correctly, not just load without error.

Known limitations
*******************

* **Axon models with a non-** ``NULL`` **** ``packed_output_buf`` **are rejected outright** by the packaging tool (``validate_shape()``): this is a compile-time array the generated model header itself declares alongside the struct, but since the deployed application no longer links that header in at all, there is nothing for the model stub to provide an address for.
* **The model stub is compiled with a standalone compiler invocation** (``nrf_axon_model_stub()``'s own ``add_custom_command()``\ s), not through Zephyr's normal per-target flag propagation - it does not automatically inherit the application's full compile flags (``-mcpu``, ``-mfpu``, ...).
  This is safe for ``nrf_axon_nn_compiled_model_s``'s own layout today (a plain, non-conditionally-compiled AAPCS struct), and ``package_model_axon.py``'s ``struct_size`` cross-check catches it regardless if that ever changes; the two Kconfig values that *do* affect whether the header even compiles (:kconfig:option:`CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE`, :kconfig:option:`CONFIG_NRF_AXON_PSUM_BUFFER_SIZE`) are passed through explicitly.
* **No runtime OTA transport.** Getting a package onto the device is a flash-only operation in this PoC; there is no over-the-air download/verify/apply flow.
* **Single model instance per partition.** Only one model package can live in a given ``model_storage*`` partition at a time; there is no A/B slot, rollback, or versioned history.
* **A model package is only valid for the exact application build it was produced alongside.** Since every app-owned pointer is resolved from that specific build's own ``zephyr.elf``, flashing a package built against a different application build (even a seemingly trivial recompile that happens to relink app-owned symbols at different addresses) is caught by the ``package_base`` check at load time, not silently accepted - but it does mean a model package cannot be prepared once and reused across arbitrary future application builds the way a Neuton package (which embeds no addresses at all) can.

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

Each sample/application additionally defines its own top-level toggle (:kconfig:option:`CONFIG_HELLO_AXON_MODEL_OTA`, :kconfig:option:`CONFIG_NRF_EDGEAI_REGRESSION_MODEL_OTA`, :kconfig:option:`CONFIG_APP_MODEL_OTA`) that ``select``\ s the ones above, plus (for :file:`applications/ww_kws`, which needs an application-owned ``persistent_vars`` RAM array per model - see "Axon packages" above) :kconfig:option:`CONFIG_APP_WW_PERSISTENT_VARS_SIZE` / :kconfig:option:`CONFIG_APP_KWS_PERSISTENT_VARS_SIZE`, which must match the size the currently bundled model's own generated header declares.

Integration pattern
**********************

A consuming application typically:

#. Declares a static, uninitialized-until-loaded model instance (``nrf_edgeai_model_neuton_t`` or ``nrf_axon_nn_compiled_model_s``) and a ``nrf_edgeai_t`` wrapping it, instead of pointing at a compiled-in model.
#. For Axon, calls ``nrf_axon_model_stub()`` from its own :file:`CMakeLists.txt` (once per model it hosts), pointed at that model's generated header and ``model_storage*`` partition - see :file:`lib/model_ota/cmake/nrf_axon_model_stub.cmake`'s own usage comment for the exact arguments.
#. Calls ``model_pkg_load_neuton()`` or ``model_pkg_load_axon()`` - against that model's own partition ID/address - at boot, and (for models that can be updated without a reboot) again periodically or on some other trigger, to (re)validate and wire up whatever currently sits in that partition.
#. Skips inference (rather than crashing) when the load fails, propagating that failure however best fits the application's own structure - for example returning an error from ``main()`` (:file:`applications/ww_kws`'s ``ww_init()``/``kws_init()``, which run once at boot before an audio-streaming loop that has no natural "retry the load" point), or retrying on the next iteration of a polling loop (:file:`samples/nrf_edgeai/regression`'s ``main()``).

See :file:`samples/axon/hello_axon/src/main.c`, :file:`samples/nrf_edgeai/regression/src/model_wiring_neuton.c` / :file:`model_wiring_axon.c`, and :file:`applications/ww_kws/src/ww/model_wiring_axon.c` / :file:`src/kws/model_wiring_axon.c` for concrete examples of this pattern - the ``ww_kws`` ones are also the reference for wiring up a model that uses op extensions and/or ``persistent_vars``.
