.. _lib_model_ota:

Model-only OTA update library
#############################

Overview
********

``lib/model_ota`` loads Neuton and Axon ML models from dedicated flash partitions at runtime, independently of the application image. Models are **linked partition images** (magic ``NEI\\0``, format version 5): the compiler/linker place the descriptor and weights at the partition base so intra-image pointers are absolute flash addresses (XIP).

Two deployment modes are supported:

**Raw partition images** (for example ``samples/multi_model`` with ``overlay-ota.conf``)
  Build ``*_model_partition.hex`` / ``*_model_image.bin`` and program each partition directly with ``nrfutil device program``. Integrity is **CRC-32/IEEE only** on the model payload. There is no transport layer in the library itself.

**MCUboot-wrapped model images** (for example :ref:`WW KWS model OTA <app_ww_kws_model_ota>`)
  Raw model images are signed with ``imgtool`` as separate MCUboot updateable images. Upload uses the normal MCUboot + MCUmgr SMP path. At boot, :c:func:`model_image_read_and_validate()` skips a leading MCUboot header (32 bytes) before validating the model payload CRC and contract.

Production update flow (raw partition flash)
********************************************

1. Build and flash firmware 1.0 (application + initial model images). Archive ``model_ota_context.json`` from the build directory alongside ``zephyr.hex``.
2. Retrain the model (same task, same I/O shape, same solution pipeline). Rebuild only the affected ``*_model_image.bin`` / ``*_model_partition.hex``.
3. Run ``check_model_compat.py`` against the archived context before release:

   - exit **0** — compatible; flash the partition hex
   - exit **2** — model exceeds firmware caps (for example ``MAX_NEURONS`` grew); ship firmware 1.1 first
   - exit **1** — incompatible (contract hash, binding, or format)

4. On boot, ``model_image_load_neuton()`` / ``model_image_load_axon()`` re-validate contract hash, caps, and (Axon) address bindings.

Production update flow (MCUboot SMP)
**********************************

1. Build and flash the full sysbuild chain (bootloader, signed application, signed model images). Archive ``model_ota_context.json`` from the build directory alongside ``zephyr.hex``. See :ref:`WW KWS model OTA <app_ww_kws_model_ota>` for provisioning with ``*_provision.hex``.
2. Retrain the model (same task, same I/O shape, same solution pipeline). Rebuild only the affected ``*_model_image.bin``; CMake produces ``*_model_mcuboot.signed.{bin,hex}`` via ``model_ota_mcuboot_sign()``.
3. Run ``check_model_compat.py`` against the archived context before release:

   - exit **0** — compatible; upload the signed model over SMP
   - exit **2** — model exceeds firmware caps (for example ``MAX_NEURONS`` grew); ship firmware 1.1 first
   - exit **1** — incompatible (contract hash, binding, or format)

4. Upload over SMP (MCUmgr ``image upload`` with the model image index), test, and reset. The application pauses inference on that model during upload.
5. On boot, ``model_image_load_neuton()`` / ``model_image_load_axon()`` re-validate contract hash, caps, and (Axon) address bindings.

Image header (format v5, 48 bytes)
**********************************

Shared envelope (offsets 0–27):

- magic ``NEI\\0``, ``format_version`` (5), ``params_type``, ``image_size``, ``model_version``
- ``contract_hash`` (off 16) — FNV-1a over the firmware ABI contract (see ``model_contract.h``)
- ``crc32`` (off 20) — CRC over the whole image with this field zeroed
- ``name`` — pointer to a NUL-terminated string in the image

Backend union (off 28, 20 bytes):

- **Neuton** (12 B used): ``model``, ``task``, ``decoded_output`` pointer
- **Axon** ``axon`` (20 B): ``model``, ``axon_packed_output_bytes``, ``persistent_vars_required``, ``binding``, ``binding_count``

When MCUboot wraps a partition, the 48-byte model header starts at ``partition_base + 32``. Partition images are still linked at that payload address (``model_ota_image_link_addr()`` in :file:`lib/model_ota/cmake/model_ota_common.cmake`).

Contract vs binding
*******************

**Contract hash** — same value in the image and compiled into the app; checked at load time.

- Neuton: task, precision, struct sizes, input/output/neuron caps, solution ID, DSP/windowing pipeline identity
- Axon: compiled-model struct size, interlayer/psum buffer Kconfig sizes, persistent-vars and packed-output requirements

**Binding table** (Axon only) — ``{name_hash, address}`` rows for each app-owned symbol the image was linked against (``nrf_axon_interlayer_buffer``, persistent vars, op extensions, ``axonpro_*``). The loader compares image addresses to the live table emitted by ``model_ota_axon_keep_refs.S``.

Host tools
**********

Under ``tools/model_ota/``:

- ``patch_image_crc.py`` — patches CRC at offset 20
- ``validate_model_image_layout.py`` — post-link build gate
- ``axon_elf.py`` — Axon probe inspect + ``PROVIDE()`` fragment from ``zephyr.elf``
- ``export_model_ota_context.py`` / ``model_ota_context.cmake`` — emit ``model_ota_context.json``
- ``check_model_compat.py`` — pre-flash compatibility verdict

CMake integration
*****************

Core wiring:

- ``model_ota_neuton_wire()`` + ``model_ota_neuton_image()`` — Neuton
- ``model_ota_axon_model()`` — pure Axon
- ``model_ota_axon_edgeai_wire()`` — Edge AI Lab wrapper + Axon backend
- ``model_ota_context_finalize()`` — export ``model_ota_context.json`` after the app links (required for in-tree model image builds on format v4+)

MCUboot helpers (optional, when ``CONFIG_BOOTLOADER_MCUBOOT`` is enabled):

- ``model_ota_mcuboot_sign()`` (:file:`lib/model_ota/cmake/model_ota_mcuboot_sign.cmake`) — signs a raw ``*_model_image.bin`` with ``imgtool`` (``--pad-header``, ``--rom-fixed`` at the partition base)
- ``model_ota_register_provision_hex()`` / ``model_ota_create_provision_hex()`` (:file:`lib/model_ota/cmake/model_ota_sysbuild.cmake`) — register signed model hex files as sysbuild flash domains and merge bootloader, app, and models into ``*_provision.hex``

Optional CMake cache variables for **out-of-tree** model partition builds against shipped
firmware (see ``tools/model_ota/README.md``):

- ``MODEL_OTA_FW_ELF`` — released ``zephyr.elf`` (Axon symbol resolution; required with context for Axon out-of-tree)
- ``MODEL_OTA_FW_CONTEXT`` — released ``model_ota_context.json`` (build-time compat check)

When either is set, in-tree context export is skipped and partition-loader wiring is omitted.
The application build is deliberately blocked (before any app source is compiled) and reports
how to build a partition image. Configure with ``west build --cmake-only``, then::

  cmake --build build/multi_model --target <name>_model_image

A default ``west build`` or ``cmake --build`` without ``--target`` fails on that block.
Axon partition images are excluded from the default build unless ``MODEL_OTA_FW_ELF`` is also
set (``axon_elf.py provide`` needs a released ELF).

SMP upload coordination
***********************

When ``CONFIG_MODEL_OTA_SMP`` is enabled, register the MCUboot updateable image index for each partition-resident model with ``model_ota_smp_init()`` (:file:`include/model_ota/model_ota_smp.h`).
Before running inference from a model partition, check ``model_ota_smp_blocks_inference()``.
After any model upload completes, reset the device before loading the new image (``model_ota_smp_is_pending_reset()``).

Known limitations
*****************

- MCUboot model updates overwrite the model partition in place.
Inference on that model is paused during SMP upload and a reset is required before running against the new image.
Check for inference blocking should be moved to different layer than the application layer (e.g. Edge AI Library, Axon driver, model OTA library).
- Axon images bind to app RAM addresses from the firmware they were linked against; binding check catches drift
- Neuton solution wrapper (DSP pipeline, decode interfaces) is not swappable — only the Neuton model payload
- If a retrained Axon model needs a new op-extension symbol the old firmware never kept, image link fails at build time

Kconfig
*******

- ``CONFIG_MODEL_OTA`` — master enable
- ``CONFIG_MODEL_OTA_NEUTON`` / ``CONFIG_MODEL_OTA_AXON`` — backends
- ``CONFIG_MODEL_OTA_SMP`` — pause inference during MCUboot SMP model uploads

See also ``tools/model_ota/README.md``, ``samples/multi_model/overlay-ota.conf``, and :ref:`WW KWS model OTA <app_ww_kws_model_ota>`.
