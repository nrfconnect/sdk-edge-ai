<!-- Copyright (c) 2026 Nordic Semiconductor ASA -->
<!-- SPDX-License-Identifier: LicenseRef-Nordic-5-Clause -->

# Model-only OTA host tools

## Summary

A Neuton or Axon model is shipped as a self-contained, **linked partition image**. The model
descriptor and data are linked at the model partition's flash base, with an 84-byte header (format
version 12, see `include/model_ota/model_image.h`) holding a direct pointer to the descriptor,
a firmware **contract hash** (offset 16), CRC-32/IEEE (offset 20), and the model's `nrf_edgeai_t`
parameter block (offset 48): its feature scaling factors and decoded-output init. That block is
shared by both backends; only a pure Axon model, having no `nrf_edgeai_t`, leaves it zeroed.

Almost all of the image is produced by the compiler/linker. These host scripts perform the work
that cannot be expressed directly in the toolchain:

- `patch_image_crc.py` - computes CRC-32/IEEE over the finished image binary (with the header's
  `crc32` field at offset 20 held at 0) and writes it back. These 4 bytes are the only
  host-written bytes in the image; the loader recomputes the CRC exactly the same way.
- `validate_model_image_layout.py` - a post-link check that fails the build if the on-flash
  header disagrees with the link: image linked at the partition base, header first, correct
  magic/format-version, image and partition sizes, model pointers, and CRC.
- `check_model_compat.py` - compares a model image against a `model_ota_context.json` (exit 0
  compatible, 1 incompatible, 2 requires firmware update). Also the build's gate on the contract
  hash: the image carries the stub's value and the context the probe's, so requiring them to agree
  is what keeps the released value equal to the one images carry. That verdict is fatal even under
  `--report-only`, which only makes the capacity verdicts a report.
- `export_model_ota_context.py` - invoked from CMake to emit `model_ota_context.json` per app
  build (partition map, caps, contract hashes, Axon symbol addresses).
- `elf_const.py` - reads a compile-time constant back out of an object. The contract hash mixes
  `sizeof` of the runtime structs, so only the compiler can fold it; the build compiles
  `lib/model_ota/src/model_ota_contract_probe.c` per slot and this reads the finished word,
  which is why no host script reimplements the hash.
- `emit_contract_slot.py` / `emit_axon_context_slot.py` - turn a slot's probe (and, for Axon, its
  generated config header) into the build-time half of its `model_ota_context.json` entry.
- `model_contract.py` - the *string* hashes the preprocessor cannot do: the solution ID mixed into
  the solution contract, and the Axon binding table's symbol names. Both are 32-bit `blake2s`
  rather than the contract hash's FNV-1a chain, since C only consumes them as literals and never
  recomputes one - and an avalanching hash is what keeps two long-shared-prefix symbol names from
  colliding into the wrong binding row. Run as `model_contract.py solution-id-hash --
  <SOLUTION_ID>`, it prints the first of those, which is how CMake obtains
  `MODEL_OTA_SOLUTION_ID_HASH`.
- `axon_elf.py` - inspects compiler-resolved Axon model metadata and resolves application
  symbols used by Axon partition images.

## In the build

The tools are wired into the OTA build; there is no manual step. Building the `multi_model`
sample with the OTA overlay produces a standalone addressed `<name>_model_partition.hex` per
model at the build directory root, plus the OTA payload `<name>_model_image.bin` under
`model_ota/<name>/`:

```bash
cd edge-ai/samples/multi_model
nrfutil toolchain-manager launch --ncs-version v3.4.0 -- \
  west build -p always -b nrf54lm20dk/nrf54lm20b/cpuapp -d build . \
  -- -DEXTRA_CONF_FILE=overlay-ota.conf
ls build/multi_model/*_model_partition.hex
ls build/multi_model/model_ota/*/*_model_image.bin
ls build/multi_model/model_ota_context.json
```

The build also emits **`model_ota_context.json`** — archive this alongside `zephyr.hex` and
`zephyr/zephyr.elf` at firmware release.

### Build directory layout

Only the release artifacts stay at the build directory root; everything else the OTA build
generates lives under `model_ota/`:

- `<name>_model_partition.hex` - addressed hex, flashed into the model's partition.
- `model_ota_context.json` - firmware context, archived at release.
- `model_ota/<name>/` - one subfolder per model: the OTA payload `<name>_model_image.bin`, the
  linked `.elf`, the raw pre-CRC `.bin`, probes, generated headers, `context_slot.json`, and the
  model's app-side static library.
- `model_ota/` - build-wide artifacts: `model_ota_context_manifest.json`,
  `model_ota_discard.ld`, and the Neuton wired sources and libraries (keyed by solution ID
  rather than by model target).

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
The compiled Axon model is partition-loaded via `model_ota_axon_model()`, same as a pure Axon
model, and the image additionally carries the solution's `nrf_edgeai_t` parameters. The wrapper
itself (windowing, pipeline, interfaces) stays compiled into the app; its
`model.instance.p_void` and its parameters are patched at runtime by
`nrf_edgeai_load_user_model_<id>()` from
`lib/model_ota/src/model_ota_axon_edgeai_wired.c.in` (the `multi_model` sample's `wakeword`,
`classif_axon`, and `regress_axon` declarations exercise this path).

## Pre-flight compatibility check

Before flashing a model built against a released firmware:

```bash
python3 tools/model_ota/check_model_compat.py \
  --context /path/to/model_ota_context.json \
  --image build/multi_model/model_ota/gear_anomaly/gear_anomaly_model_image.bin \
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
  `include/model_ota/model_contract.h` (the one definition of the contract hash),
  `lib/model_ota/src/model_ota_contract_probe.c` (how the host gets its value),
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
