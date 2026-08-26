.. _lib_model_ota:

Model-only OTA update library
#############################

Overview
********

``lib/model_ota`` loads Neuton and Axon ML models from dedicated flash partitions at runtime, independently of the application image. Models are **linked partition images** (magic ``NEI\\0``, format version 10): the compiler/linker place the descriptor and weights at the partition base so intra-image pointers are absolute flash addresses (XIP).

There is **no runtime OTA transport** in this library: getting an image onto the device is a flash-only operation (for example ``nrfutil device program`` on a partition ``.hex``). Integrity is **CRC-32/IEEE only** — there is no signature or authenticity check and no MCUboot involvement.

Production update flow
**********************

1. Build and flash firmware 1.0 (application + initial model images). Archive ``model_ota_context.json`` from the build directory alongside ``zephyr.hex``.
2. Retrain the model (same task, same I/O shape, same solution pipeline). Rebuild only the affected ``*_model_image.bin`` / ``*_model_partition.hex``.
3. Run ``check_model_compat.py`` against the archived context before release:

   - exit **0** — compatible; flash the partition hex
   - exit **2** — model exceeds firmware caps (for example ``MAX_NEURONS`` grew); ship firmware 1.1 first
   - exit **1** — incompatible (contract hash, binding, or format)

4. On boot, ``model_image_load_neuton()`` / ``model_image_load_axon()`` re-validate contract hash, caps, and (Axon) address bindings.

Image header (format v10, 80 bytes)
***********************************

Shared envelope (offsets 0–27):

- magic ``NEI\\0``, ``format_version`` (9), ``params_type``, ``image_size``, ``model_version``
- ``contract_hash`` (off 16) — FNV-1a over the firmware ABI contract (see ``model_contract.h``)
- ``crc32`` (off 20) — CRC over the whole image with this field zeroed
- ``name`` — pointer to a NUL-terminated string in the image

Backend union (off 28, 20 bytes):

- **Neuton** (4 B used): ``model``; the rest of the slot is zeroed
- **Axon** ``axon`` (20 B): ``model``, ``axon_packed_output_bytes``, ``persistent_vars_required``, ``binding``, ``binding_count``

Runtime parameters (off 48, 32 bytes): ``edgeai_params``, a ``struct model_image_edgeai_params`` holding the model's share of ``nrf_edgeai_t`` — the values the runtime keeps outside the backend model descriptor, and which therefore have to travel with the model:

- ``scale`` — a ``nrf_edgeai_input_scale_t`` / ``nrf_edgeai_features_meta_t`` union holding the baked scaling factors (and, for a DSP pipeline, the extraction arguments)
- ``decoded_output`` — the baked ``NN_DECODED_OUTPUT_INIT``, including the output decode meta
- ``scale_num``, ``scale_elem_size`` — scaling array geometry, checked against the application's expectation

The block sits outside the backend union because what a solution keeps in ``nrf_edgeai_t`` does not depend on whether its network is Neuton or Axon. A ``scale_num`` of 0 means the image carries no parameters and the application keeps its compiled-in values; the rest of the block is then zeroed too. That is the case for a **pure Axon** model, which has no ``nrf_edgeai_t`` at all — an Axon-backed Edge AI Lab solution carries its parameters like any Neuton one.

Everything is stored in the runtime's own types, so ``model_image_bind_edgeai_params()`` applies it by plain assignment. It is called by the wired translation unit right after ``model_image_load_neuton()`` / ``model_image_load_axon()`` succeeds, not by the loader itself.

A solution scales once on the way into the network, so exactly one of ``INPUT_FEATURES_SCALE_MIN/MAX`` and ``EXTRACTED_FEATURES_SCALE_MIN/MAX`` belongs to the model and travels in the image; the other stage's factors, if any, stay compiled into the application. ``model_ota_scale_select.h`` makes that choice for both the image stub and the wired application translation unit, so the two agree by construction — for either backend. Which one the loader reads back is not recorded in the image: it follows from the solution, so it is taken from ``nrf_edgeai_t.p_dsp`` (a context with a DSP pipeline scales its extracted features, one without scales its raw input features).

Contract vs binding
*******************

**Contract hash** — same value in the image and compiled into the app; checked at load time.

- Neuton: task, precision, struct sizes, input/output/neuron caps, solution ID, DSP/windowing pipeline identity
- Axon: compiled-model struct size, interlayer/psum buffer Kconfig sizes, persistent-vars and packed-output requirements

The image itself carries no task byte: for Neuton the task is already covered by the contract hash. The Axon contract hash does not cover the task or the solution identity yet, so two Axon models that agree on those sizes are currently interchangeable as far as the loader can tell.

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

- ``model_ota_neuton_wire()`` + ``model_ota_neuton_image()`` — Neuton
- ``model_ota_axon_model()`` — pure Axon
- ``model_ota_axon_edgeai_wire()`` — Edge AI Lab wrapper + Axon backend

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

Known limitations
*****************

- No OTA transport, signing, A/B slots, or rollback
- Axon images bind to app RAM addresses from the firmware they were linked against; binding check catches drift
- The solution wrapper (DSP pipeline, decode interfaces) is not swappable — only the model payload and the ``nrf_edgeai_t`` parameters that travel with it
- ``model_image_bind_edgeai_params()`` trusts the image it is handed: it assumes the backend loader validated it first, and only cross-checks the scaling geometry
- If a retrained Axon model needs a new op-extension symbol the old firmware never kept, image link fails at build time

Kconfig
*******

- ``CONFIG_MODEL_OTA`` — master enable
- ``CONFIG_MODEL_OTA_NEUTON`` / ``CONFIG_MODEL_OTA_AXON`` — backends

See also ``tools/model_ota/README.md`` and ``samples/multi_model/overlay-ota.conf``.
