.. _lib_model_ota:

Model-only OTA update library
#############################

Overview
********

``lib/model_ota`` loads Neuton and Axon ML models from dedicated flash partitions at runtime, independently of the application image. Models are **MCUboot-wrapped linked partition images**: ``imgtool`` prepends a 32-byte MCUboot header at the partition base, and the model payload (magic ``NEI\\0``, format version 12) is linked at ``partition_base + 32`` so intra-image pointers are absolute flash addresses (XIP).

Production update flow
**********************

1. Build and flash the full sysbuild chain (bootloader, signed application, signed model images). Archive ``model_ota_context.json`` from the build directory alongside ``zephyr.hex``. See :ref:`WW KWS model OTA <app_ww_kws_model_ota>` for provisioning with ``*_provision.hex``.
2. Retrain the model (same task, same I/O shape, same solution pipeline). Rebuild only the affected ``*_model_image.bin``; CMake produces ``*_model_mcuboot.signed.{bin,hex}`` via ``model_ota_mcuboot_sign()``.
3. Run ``check_model_compat.py`` against the archived context before release:

   - exit **0** — compatible; upload the signed model over SMP
   - exit **2** — model exceeds firmware caps (for example ``NEURONS_CAP`` grew); ship firmware 1.1 first
   - exit **1** — incompatible (contract hash, binding, or format)

4. Upload over SMP (MCUmgr ``image upload`` with the model image index), then reset. The built-in inference guard blocks all OTA-managed Edge AI inference device-wide during upload and until reset.
5. On boot, ``model_image_load_neuton()`` / ``model_image_load_axon()`` skip the MCUboot header, then re-validate contract hash, caps, and (Axon) address bindings.

Image header (format v12, 84 bytes)
***********************************

Shared envelope (offsets 0–27):

- magic ``NEI\\0``, ``format_version`` (12), ``params_type``, ``image_size``, ``model_version``
- ``contract_hash`` (off 16) — FNV-1a over the firmware ABI contract (see ``model_contract.h``)
- ``crc32`` (off 20) — CRC over the whole image with this field zeroed
- ``name`` — pointer to a NUL-terminated string in the image

Backend union (off 28, 20 bytes):

- **Neuton** (4 B used): ``model``; the rest of the slot is zeroed
- **Axon** ``axon`` (20 B): ``model``, ``axon_packed_output_bytes``, ``persistent_vars_required``, ``binding``, ``binding_count``

Runtime parameters (off 48, 36 bytes): ``edgeai_params``, a ``struct model_image_edgeai_params`` holding the model's share of ``nrf_edgeai_t`` — the values the runtime keeps outside the backend model descriptor, and which therefore have to travel with the model:

- ``scale`` — a ``nrf_edgeai_input_scale_t`` / ``nrf_edgeai_features_meta_t`` union holding the baked scaling factors (and, for a DSP pipeline, the extraction arguments)
- ``decoded_output`` — the baked ``NN_DECODED_OUTPUT_INIT``, including the output decode meta
- ``p_extraction_mask`` — the solution's ``FEATURES_EXTRACTION_MASK[]``, carried for *verification* rather than to be applied (NULL without a DSP pipeline)
- ``scale_num``, ``scale_elem_size`` — scaling array geometry, covered by the contract hash

The block sits outside the backend union because what a solution keeps in ``nrf_edgeai_t`` does not depend on whether its network is Neuton or Axon. A ``scale_num`` of 0 means the image carries no parameters and the application keeps its compiled-in values; the rest of the block is then zeroed too. That is the case for a **pure Axon** model, which has no ``nrf_edgeai_t`` at all — an Axon-backed Edge AI Lab solution carries its parameters like any Neuton one.

Everything is stored in the runtime's own types, so ``model_image_bind_edgeai_params()`` applies it by plain assignment. It is called by the wired translation unit right after ``model_image_load_neuton()`` / ``model_image_load_axon()`` succeeds, not by the loader itself. For a solution with a DSP pipeline it first compares ``p_extraction_mask`` word by word against the application's ``nrf_edgeai_t.p_dsp->features.p_masks`` and applies nothing on a mismatch, reporting which input feature's mask diverged.

A solution scales once on the way into the network, so exactly one of ``INPUT_FEATURES_SCALE_MIN/MAX`` and ``EXTRACTED_FEATURES_SCALE_MIN/MAX`` belongs to the model and travels in the image; the other stage's factors, if any, stay compiled into the application. ``model_ota_scale_select.h`` makes that choice for both the image stub and the wired application translation unit, so the two agree by construction — for either backend. Which one the loader reads back is not recorded in the image: it follows from the solution, so it is taken from ``nrf_edgeai_t.p_dsp`` (a context with a DSP pipeline scales its extracted features, one without scales its raw input features).

When MCUboot wraps a partition, the model header starts at ``partition_base + 32``. Partition images are linked at that payload address (``model_ota_image_link_addr()`` in :file:`lib/model_ota/cmake/model_ota_common.cmake`). Do not flash the unsigned ``*_model_partition.hex`` into a model slot; use the signed ``*_model_mcuboot.signed.hex`` output from ``model_ota_mcuboot_sign()``.

Contract vs binding
*******************

**Contract hash** — same value in the image and compiled into the app; checked at load time.

Only *identity* invariants are hashed: values that must be equal or the image is meaningless. Every flavor starts from the same envelope — format version and the partition base the image was linked at (which ties an image to one slot, since all its pointers are absolute flash addresses).

The base folded into the hash is ``MODEL_OTA_IMAGE_LINK_BASE``: CMake passes the same ``dt_reg_addr()`` literal to the contract probe, the partition-image stub, wired loaders (for hashing), and the standalone image link (``--defsym`` in ``model_image.ld``). Wired loaders also take a devicetree nodelabel and ``BUILD_ASSERT`` that ``MODEL_OTA_PARTITION_ADDR()`` matches ``MODEL_OTA_IMAGE_LINK_BASE`` before dereferencing the mapped partition at runtime.

- **Edge AI Lab / Neuton** (``MODEL_OTA_CONTRACT_HASH_EDGEAI_NEUTON``): envelope, weight precision, descriptor/meta struct sizes, then the solution contract
- **Edge AI Lab / Axon** (``MODEL_OTA_CONTRACT_HASH_EDGEAI_AXON``): envelope, the same driver ABI, then the solution contract
- **raw Axon** (``MODEL_OTA_CONTRACT_HASH_AXON``): envelope and the driver ABI — compiled-model struct size, interlayer and psum Kconfig sizes

The *solution contract* is shared by both ``nrf_edgeai_t``-wrapped flavors: solution ID, Lab runtime version, task, output count, decoded-output and parameter-block struct sizes, scaling geometry, input feature type and count, window size/shift/subwindows, extracted-feature count, ``MODEL_USES_AS_INPUT_MASK``, and a DSP digest.

That DSP digest (``MODEL_OTA_CONTRACT_HASH_DSP``, 0 for a solution without a feature pipeline) covers the extent and element width of ``FEATURES_EXTRACTION_ARGUMENTS`` and the FFT geometry — ``DSP_AMPLITUDE_SPECTRUM_LEN``, ``DSP_RFFT_LEN`` and the bit-reversal table length, all 0 without a frequency-domain pipeline. The FFT and twiddle *table contents* are not hashed: they are a pure function of those lengths. ``FEATURES_EXTRACTION_MASK`` is not hashed either — the preprocessor cannot fold an array, so it travels in the image and is compared at load time instead (see ``p_extraction_mask`` above).

There is no backend tag in the hash. The flavors differ in the *sequence* of chunks they fold in — a wrapped solution adds sixteen more than a raw Axon model — so an FNV-1a chain plus the final avalanche separates them without one, and each loader independently checks ``params_type``. A pure Axon image can therefore never be accepted by a wrapped solution's slot. That matters: such an image carries no ``edgeai_params``, while an OTA-wired application has discarded its own compiled-in copy.

The solution ID hashed here is the ``SOLUTION_ID`` the CMake helper was called with, not the generated source's ``EDGEAI_LAB_SOLUTION_ID_STR``, so both sides of an update derive it from the value the build was configured with.

**One implementation, the compiler's.** Every value above is folded by the C macros — nothing recomputes the hash on the host. Since the mix includes ``sizeof`` of the runtime structs, the preprocessor cannot evaluate it and a host reimplementation would need hand-maintained struct sizes that go stale behind a runtime header change. Instead the build compiles ``lib/model_ota/src/model_ota_contract_probe.c`` once per slot — a throwaway translation unit built with the application's own flags and linked into neither the firmware nor the image — and the host reads the finished word out of that object (``tools/model_ota/elf_const.py``). That is the value ``model_ota_context.json`` carries. The image's own copy is baked by the flavor's stub, a separate translation unit, and ``check_model_compat.py`` requires the two to agree — which is what keeps the value released to the field equal to the one images actually carry, and is fatal even in the in-tree ``--report-only`` build report. The only hashing left on the host is over *strings*, which the preprocessor genuinely cannot do: the solution ID above and the Axon binding table's symbol names. Those two are a 32-bit ``blake2s`` rather than the FNV-1a chain — C only ever consumes them as literals, never recomputing one, so they are free of the constraint that shaped the contract hash, and an avalanching hash is what keeps two Axon symbol names sharing a long prefix from colliding into the wrong binding row.

**Capacities are deliberately not hashed.** The Neuton neuron scratch capacity and the Axon persistent-vars and packed-output caps have a "required <= provided" relation, so they stay as header fields checked by inequality. Hashing them would collapse the "model outgrew this firmware" verdict (exit 2) into an opaque contract mismatch.

.. note::

   None of these hashes is a security control — not the contract hash, whose FNV-1a is unkeyed, and not the unkeyed ``blake2s`` over the strings either. Nor is the image CRC an authenticity check. All of them guard against accidents. Authenticity enforcement by MCUboot will be added in future commits.

**Binding table** (Axon only) — one device-wide ``{name_hash, address}`` table for every app-owned symbol any Axon image was linked against (``nrf_axon_interlayer_buffer``, persistent vars, op extensions, ``axonpro_*``). ``axon_elf.py binding-table`` unions the per-slot keep lists at finalize; ``model_ota_axon_keep_refs.S`` emits ``model_ota_axon_binding_table``, which ``model_image_load_axon()`` consults directly.

Host tools
**********

Under ``tools/model_ota/``:

- ``patch_image_crc.py`` — patches CRC at offset 20
- ``validate_model_image_layout.py`` — post-link build gate
- ``elf_const.py`` — reads the compiler-folded contract hash out of a slot's contract probe
- ``axon_elf.py`` — Axon probe inspect, device-wide binding-table merge, and ``PROVIDE()`` fragment from ``zephyr.elf``
- ``export_model_ota_context.py`` / ``model_ota_context.cmake`` — emit ``model_ota_context.json``
- ``check_model_compat.py`` — pre-flash compatibility verdict, and the build's contract-hash gate

CMake integration
*****************

Include ``lib/model_ota/cmake/model_ota.cmake`` once, then:

- ``model_ota_edgeai_neuton_model()`` — Edge AI Lab solution, Neuton backend
- ``model_ota_edgeai_axon_model()`` — Edge AI Lab solution, Axon backend
- ``model_ota_axon_model()`` — raw Axon model (no ``nrf_edgeai_t`` wrapper)
- ``model_ota_finalize()`` — after all slot declarations; merges the Axon binding table and
  exports ``model_ota_context.json`` once the application links

MCUboot helpers:

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

- MCUboot model updates overwrite the model partition in place. Inference on that model is paused during SMP upload and a reset is required before running against the new image (guard stays BLOCKED until reset).
- The loaders do not range-check the image's baked pointers: the partition base they were linked at is part of the contract hash, and containment is gated at build time by ``validate_model_image_layout.py``. A crafted image with a valid CRC and contract hash is therefore not contained — authenticity is MCUboot's job in the final solution
- Axon images bind to app RAM addresses from the firmware they were linked against; binding check catches drift
- The solution wrapper (DSP pipeline, decode interfaces) is not swappable — only the model payload and the ``nrf_edgeai_t`` parameters that travel with it
- ``model_image_bind_edgeai_params()`` trusts the image it is handed: apart from comparing the extraction mask, it assumes the backend loader validated it first, and that the contract hash already covered the scaling geometry
- If a retrained Axon model needs a new op-extension symbol the old firmware never kept, image link fails at build time
- A contract mismatch surfaces once the slot's contract probe has compiled, i.e. during the build rather than at CMake configure time, since that is when the compiler's value first exists
- A Lab release that changes what an extraction function *computes* while keeping the same ``FEATURES_EXTRACTION_MASK`` is caught only by ``EDGEAI_RUNTIME_VERSION_COMBINED`` — a version bump being trusted, not a digest of the implementation

Kconfig
*******

- ``CONFIG_MODEL_OTA`` — master enable (``CONFIG_BOOTLOADER_MCUBOOT`` or ``CONFIG_MODEL_OTA_TESTING``)
- ``CONFIG_MODEL_OTA_TESTING`` — sample/dev mode: empty 32-byte header pad, direct partition hex flash
- ``CONFIG_MODEL_OTA_NEUTON`` / ``CONFIG_MODEL_OTA_AXON`` — backends
- ``CONFIG_MODEL_OTA_SMP`` — SMP upload callbacks and inference guard integration
- ``CONFIG_MODEL_OTA_GUARD_DRAIN_TIMEOUT_MS`` — reader-drain timeout at upload start
- ``CONFIG_MODEL_OTA_SMP_DRAIN_RETRY_WINDOW_MS`` — how long inference stays blocked after a
  deferred first chunk, waiting for the client to retry the upload

See also ``tools/model_ota/README.md``, ``samples/multi_model/overlay-ota.conf``, and :ref:`WW KWS model OTA <app_ww_kws_model_ota>`.
