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

4. Upload over SMP (MCUmgr ``image upload`` with the model image index), then reset. The built-in inference guard blocks all OTA-managed Edge AI inference device-wide during upload and until reset.
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

When ``CONFIG_MODEL_OTA_SMP`` is enabled, each OTA-wired loader
(``nrf_edgeai_load_user_model_<solution_id>()``) registers its MCUboot image index through
``model_ota_smp_register()`` before reading the model image, so uploads are coordinated even
when the partition holds an invalid image. A registration failure other than ``-EALREADY``
fails the load. Image indices are derived from the ``nordic,mcuboot-image`` bootchain in
devicetree (:file:`include/model_ota/model_ota_partition.h`).

Inference guard (``CONFIG_MODEL_OTA``)
**************************************

When model OTA is enabled, ``lib/model_ota`` provides a **device-wide inference guard**.
Edge AI wired models are protected automatically through the runtime gate hooks.
Direct Axon driver use and other model flash access require explicit
``model_ota_guard_acquire()`` / ``model_ota_guard_release()`` pairs in
application or library code.

Global state machine (``model_ota_guard.h``):

- **READY** — model access and inference permitted (initial state at boot)
- **BLOCKED** — SMP upload reserved, active or finished/aborted; all OTA-managed
  inference is refused

Once model flash is modified, only a device reset returns the guard to READY.
An upload that is given up on before any erase or write releases the guard with
``model_ota_guard_abort_update()``, and inference continues on the unchanged
model. Updating any one model partition blocks inference device-wide.

``model_ota_smp_is_pending_reset()`` is true only after the upload session ends
(success or abort), not while the transfer is still in progress.

A single global reader count tracks in-flight model access. SMP upload waits
for readers to reach zero before erasing flash. The wait uses one absolute
deadline (``CONFIG_MODEL_OTA_GUARD_DRAIN_TIMEOUT_MS``): each wakeup subtracts
elapsed time rather than restarting the full timeout, so a steady stream of
short inferences cannot extend the wait beyond the configured limit.

Enforcement:

- **Edge AI runtime** — ``nrf_edgeai_run_inference()`` checks
  ``nrf_edgeai_t.is_ota_managed`` and returns ``NRF_EDGEAI_ERR_UNAVAILABLE``
  when global state is not READY. For OTA-managed contexts, the guard overrides
  weak ``nrf_edgeai_guard_inference_session_*()`` hooks. The reader is held from
  session begin through ``run_inference`` and ``propagate_outputs`` (Axon
  dequantize reads quant fields from the XIP model struct). ``decode_outputs``
  runs after session end — it only touches RAM outputs and firmware-side decode
  metadata.
- **Application / library code** — call ``model_ota_guard_acquire()`` before
  touching model-linked data outside ``nrf_edgeai_run_inference()`` (for example
  Axon quantization fields, direct ``nrf_axon_nn_model_infer_*()``, or other
  model flash metadata). Release when done; for async Axon inference, hold until
  the completion callback returns.

Upload arbitration (``model_ota_smp.c``)
========================================

All registered slots share one state machine, because the guard is device-wide
and MCUmgr keeps a single upload session. It is driven by ``img_mgmt`` events:

- **DFU_CHUNK** at offset 0 for a model image calls
  ``model_ota_guard_begin_update()``. On success the chunk is authorised and the
  guard is held; further chunks of that transfer need no drain.
- **DFU_STARTED** confirms the transfer really began and is the point where the
  upload-active notification fires. ``img_mgmt`` erases and writes only after
  this event, so a **DFU_STOPPED** before it (for example another upload-check
  handler rejecting the same chunk) releases the guard.
- **DFU_PENDING** (transfer complete) and **DFU_STOPPED** after DFU_STARTED both
  leave the guard BLOCKED until reset and make
  ``model_ota_smp_is_pending_reset()`` true.
- **DFU_CHUNK** at offset 0 for a *non-model* image supersedes any model session:
  a reservation that never wrote flash is released, a transfer that did write
  stays blocked until reset.

Deferred first chunk
--------------------

If readers do not drain within ``CONFIG_MODEL_OTA_GUARD_DRAIN_TIMEOUT_MS``, the
first chunk is rejected with ``MGMT_ERR_EBUSY`` but the guard is **left BLOCKED**
for ``CONFIG_MODEL_OTA_SMP_DRAIN_RETRY_WINDOW_MS``. New inference sessions are
refused during that window, so the in-flight ones finish. ``img_mgmt`` discards
the upload session together with the rejection, so the client's retry arrives as
another chunk 0 and is accepted by a no-drain re-check
(``model_ota_guard_retry_update()``).

The guard is released and inference resumes when the retry still finds model
access in flight, or when no retry arrives within the window. Nothing was
erased or written in either case, so the device keeps running the current model.

``model_ota_smp_is_pending_reset()`` tracks upload completion for application
policy (LEDs, reboot prompts).

Applications may still idle after upload until reset (operator policy); the
guard ensures no model access runs against erased or unverified flash in the
meantime.

Known limitations
*****************

- MCUboot model updates overwrite the model partition in place; a device reset is still required before running against the new image (guard stays BLOCKED until reset).
- Axon images bind to app RAM addresses from the firmware they were linked against; binding check catches drift
- Neuton solution wrapper (DSP pipeline, decode interfaces) is not swappable — only the Neuton model payload
- If a retrained Axon model needs a new op-extension symbol the old firmware never kept, image link fails at build time

Kconfig
*******

- ``CONFIG_MODEL_OTA`` — master enable
- ``CONFIG_MODEL_OTA_NEUTON`` / ``CONFIG_MODEL_OTA_AXON`` — backends
- ``CONFIG_MODEL_OTA_SMP`` — SMP upload callbacks and inference guard integration
- ``CONFIG_MODEL_OTA_GUARD_DRAIN_TIMEOUT_MS`` — reader-drain timeout at upload start
- ``CONFIG_MODEL_OTA_SMP_DRAIN_RETRY_WINDOW_MS`` — how long inference stays blocked after a
  deferred first chunk, waiting for the client to retry the upload

See also ``tools/model_ota/README.md``, ``samples/multi_model/overlay-ota.conf``, and :ref:`WW KWS model OTA <app_ww_kws_model_ota>`.
