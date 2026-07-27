<!-- Copyright (c) 2026 Nordic Semiconductor ASA -->
<!-- SPDX-License-Identifier: LicenseRef-Nordic-5-Clause -->

# Model-only OTA host tools

## Summary

A Neuton or Axon model is shipped as a self-contained, **linked partition image**. The model
descriptor and data are linked at the model partition's flash base, with a 48-byte header (format
version 5, see `include/model_ota/model_image.h`) holding a direct pointer to the descriptor,
a firmware **contract hash** (offset 16), and CRC-32/IEEE (offset 20).

Almost all of the image is produced by the compiler/linker. These host scripts perform the work
that cannot be expressed directly in the toolchain:

- `patch_image_crc.py` - computes CRC-32/IEEE over the finished image binary (with the header's
  `crc32` field at offset 20 held at 0) and writes it back. These 4 bytes are the only
  host-written bytes in the image; the loader recomputes the CRC exactly the same way.
- `validate_model_image_layout.py` - a post-link check that fails the build if the on-flash
  header disagrees with the link: image linked at the partition base, header first, correct
  magic/format-version, image and partition sizes, model pointers, contract hash, and CRC.
- `check_model_compat.py` - compares a model image against a released `model_ota_context.json`
  (exit 0 compatible, 1 incompatible, 2 requires firmware update).
- `export_model_ota_context.py` - invoked from CMake to emit `model_ota_context.json` per app
  build (partition map, caps, contract hashes, Axon symbol addresses).
- `model_contract.py` - shared FNV-1a contract hash helpers for host tools.
- `axon_elf.py` - inspects compiler-resolved Axon model metadata and resolves application
  symbols used by Axon partition images.

## In the build

The tools are wired into the OTA build; there is no manual step. Building the `multi_model`
sample with the OTA overlay produces a standalone `<name>_model_partition.hex` (addressed) plus
`<name>_model_image.bin` per model under the build dir:

```bash
cd edge-ai/samples/multi_model
nrfutil toolchain-manager launch --ncs-version v3.4.0 -- \
  west build -p always -b nrf54lm20dk/nrf54lm20b/cpuapp -d build . \
  -- -DEXTRA_CONF_FILE=overlay-ota.conf
ls build/multi_model/*_model_partition.hex build/multi_model/*_model_image.bin
ls build/multi_model/model_ota_context.json
```

The build also emits **`model_ota_context.json`** — archive this alongside `zephyr.hex` and
`zephyr/zephyr.elf` at firmware release.

### Out-of-tree model partition rebuild

Rebuild a single partition image against **already shipped** firmware without linking a new app.
Configure first, then build one partition target explicitly:

```bash
west build -p always -b nrf54lm20dk/nrf54lm20b/cpuapp -d build . --cmake-only \
  -- -DEXTRA_CONF_FILE=overlay-ota.conf \
     -DMODEL_OTA_FW_ELF=/path/to/released/zephyr.elf \
     -DMODEL_OTA_FW_CONTEXT=/path/to/released/model_ota_context.json

cmake --build build/multi_model --target gesture_class_model_image
```

| Variable | File | Used for |
|----------|------|----------|
| `MODEL_OTA_FW_ELF` | `zephyr.elf` | Axon image link (`axon_elf.py provide` — app RAM symbol addresses) |
| `MODEL_OTA_FW_CONTEXT` | `model_ota_context.json` | Build-time `check_model_compat.py` (`--report-only`) |

**Axon** models require **both** variables when building out-of-tree. **Neuton** partition images
do not use the ELF (pass only `MODEL_OTA_FW_CONTEXT` to skip in-tree context export).

When either variable is set, CMake skips `model_ota_context` export and omits app partition-loader
wiring. The application build (`app`, `zephyr.elf`) is deliberately blocked and prints how to
build a partition image instead. Use `cmake --build build/multi_model --target <name>_model_image`
— a plain `west build` or default `cmake --build` without `--target` fails on that block.
Axon `*_model_image` targets are excluded from the default build unless `MODEL_OTA_FW_ELF` is set
(because `axon_elf.py provide` has no symbol source without a released ELF).

Neuton per-image build steps live in `lib/model_ota/cmake/model_ota_neuton_image.cmake`
(compile a model stub, link at the partition base with `lib/model_ota/linker/model_image.ld`,
`objcopy` the `.model_image` section, patch CRC, validate, emit the addressed hex). Neuton
app-image payload discard and partition loaders live in `lib/model_ota/cmake/model_ota_neuton.cmake`
(`configure_file` from `lib/model_ota/src/model_ota_neuton_wired.c.in`).

Axon per-image build steps and app wiring use one `model_ota_axon_model()` declaration in
`lib/model_ota/cmake/model_ota_axon.cmake`. By default the linked image's optional
`packed_output_buf` field is NULL and no app RAM is spent on it; pass
`ALLOCATE_PACKED_OUTPUT` to allocate app-owned storage and wire it into the image for
models that require a dedicated packing buffer (the `multi_model` sample's `person_det`
declaration uses this option, so both code paths are exercised by its OTA build).

A "Nordic EdgeAI Lab" solution exported for the Axon backend (a `nrf_edgeai_t` wrapper -
input windowing, DSP feature pipeline, decode interfaces - around a compiled Axon model) uses
`model_ota_axon_edgeai_wire()` in `lib/model_ota/cmake/model_ota_axon_edgeai.cmake` instead.
Only the compiled Axon model is partition-loaded (via `model_ota_axon_model()`, same as a pure
Axon model); the wrapper stays compiled into the app, and its `model.instance.p_void` is patched
at runtime by `nrf_edgeai_load_user_model_<id>()` from
`lib/model_ota/src/model_ota_axon_edgeai_wired.c.in` (the `multi_model` sample's `wakeword`,
`classif_axon`, and `regress_axon` declarations exercise this path).

## Pre-flight compatibility check

Before flashing a model built against a released firmware:

```bash
python3 tools/model_ota/check_model_compat.py \
  --context /path/to/model_ota_context.json \
  --image build/multi_model/gear_anomaly_model_image.bin \
  --slot gear_anomaly
```

For Axon models, also pass `--elf` pointing at the firmware ELF used when the context was
exported (or the current build's `zephyr.elf` if in-tree).

## Flashing (separate from the app)

The app (`zephyr.hex`) and each model partition are flashed independently. Program one model
image into its partition without disturbing the app or the other partitions:

```bash
nrfutil device program --firmware gear_anomaly_model_partition.hex \
  --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM
```

## References

- Image format and loader: `include/model_ota/model_image.h`,
  `include/model_ota/model_contract.h`,
  `lib/model_ota/model_image_neuton.c`, `lib/model_ota/model_image_axon.c`
- Production flow doc: `doc/libraries/model_ota.rst`
- Context export: `lib/model_ota/cmake/model_ota_context.cmake`
- Build wiring: `lib/model_ota/cmake/model_ota_neuton_image.cmake`,
  `lib/model_ota/cmake/model_ota_neuton.cmake`, `lib/model_ota/src/model_ota_neuton_wired.c.in`,
  `lib/model_ota/src/model_ota_neuton_image_stub.c`,
  `lib/model_ota/src/model_ota_stub_macros.h`, `lib/model_ota/linker/model_image.ld`
- Axon wiring: `lib/model_ota/cmake/model_ota_axon.cmake`,
  `tools/model_ota/axon_elf.py`
- Edge AI Lab / Axon-backend wiring: `lib/model_ota/cmake/model_ota_axon_edgeai.cmake`,
  `lib/model_ota/src/model_ota_axon_edgeai_wired.c.in`,
  `include/model_ota/model_ota_axon_edgeai.h`
