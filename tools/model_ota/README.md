<!-- Copyright (c) 2026 Nordic Semiconductor ASA -->
<!-- SPDX-License-Identifier: LicenseRef-Nordic-5-Clause -->

# Model-only OTA host tools

## Summary

A Neuton or Axon model is shipped as a self-contained, **MCUboot-wrapped linked partition image**.
`imgtool` prepends a 32-byte MCUboot header at the partition base; the model descriptor and data
are linked at `partition_base + 32`, with an 84-byte header (format version 12, see
`include/model_ota/model_image.h`) holding a direct pointer to the descriptor, a firmware
**contract hash** (offset 16), CRC-32/IEEE (offset 20), and the model's `nrf_edgeai_t` parameter
block (offset 48): its feature scaling factors and decoded-output init. That block is shared by
both backends; only a pure Axon model, having no `nrf_edgeai_t`, leaves it zeroed.

Almost all of the image is produced by the compiler/linker. These host scripts perform the work
that cannot be expressed directly in the toolchain:

- `patch_image_crc.py` - computes CRC-32/IEEE over the finished image binary (with the header's
  `crc32` field at offset 20 held at 0) and writes it back. These 4 bytes are the only
  host-written bytes in the image; the loader recomputes the CRC exactly the same way.
- `validate_model_image_layout.py` - a post-link check that fails the build if the on-flash
  header disagrees with the link: image linked at `partition_base + 32`, header first, correct
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
- `emit_context_slot.py` - turn a slot's probe (and, for Axon, its generated config header)
  into the build-time half of its `model_ota_context.json` entry.
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

Production builds use MCUboot signing (see :ref:`WW KWS model OTA <app_ww_kws_model_ota>`).

For development, enable ``CONFIG_MODEL_OTA_TESTING`` (``samples/multi_model/overlay-ota.conf``):
models are linked at ``partition_base + 32`` and ``*_model_partition.hex`` is addressed at that
payload offset via ``objcopy`` (no MCUboot signing).

```bash
cd edge-ai/samples/multi_model
nrfutil toolchain-manager launch --ncs-version v3.4.0 -- \
  west build -p always -b nrf54lm20dk/nrf54lm20b/cpuapp -d build . \
  -- -DEXTRA_CONF_FILE=overlay-ota.conf
ls build/multi_model/*_model_partition.hex
```

The build also emits **`model_ota_context.json`** — archive this alongside `zephyr.hex` and
`zephyr/zephyr.elf` at firmware release.

### Build directory layout

Only the release artifacts stay at the build directory root; everything else the OTA build
generates lives under `model_ota/`:

- `<name>_model_mcuboot.signed.{bin,hex}` - MCUboot-wrapped model image for SMP upload or
  provisioning.
- `model_ota_context.json` - firmware context, archived at release.
- `model_ota/<name>/` - one subfolder per model: the OTA payload `<name>_model_image.bin`, the
  linked `.elf`, the raw pre-CRC `.bin`, probes, generated headers, `context_slot.json`, and the
  model's app-side static library.

### Out-of-tree model partition rebuild

Rebuild a single partition image against **already shipped** firmware without linking a new app.
Configure first, then build one partition target explicitly:

```bash
west build -p always -b nrf54lm20dk/nrf54lm20b/cpuapp -d build . --cmake-only \
  -- -DMODEL_OTA_FW_ELF=/path/to/released/zephyr.elf \
     -DMODEL_OTA_FW_CONTEXT=/path/to/released/model_ota_context.json

cmake --build build/ww_kws --target ww_model_image
```

| Variable | File | Used for |
|----------|------|----------|
| `MODEL_OTA_FW_ELF` | `zephyr.elf` | Axon image link (`axon_elf.py provide` — app RAM symbol addresses) |
| `MODEL_OTA_FW_CONTEXT` | `model_ota_context.json` | Build-time `check_model_compat.py` (`--report-only`) |

**Neuton** partition images do not use the ELF (pass only `MODEL_OTA_FW_CONTEXT` to skip
in-tree context export). **Axon** models require **both** variables when building out-of-tree.

When either variable is set, CMake skips `model_ota_context` export and omits app partition-loader
wiring. The application build (`app`, `zephyr.elf`) is deliberately blocked and prints how to
build a partition image instead. Use `cmake --build <build-dir> --target <name>_model_image`
— a plain `west build` or default `cmake --build` without `--target` fails on that block.
Axon `*_model_image` targets are excluded from the default build unless `MODEL_OTA_FW_ELF` is set
(because `axon_elf.py provide` has no symbol source without a released ELF).

`model_ota_edgeai_neuton_model()` in `lib/model_ota/cmake/model_ota_edgeai_neuton.cmake`,
`model_ota_edgeai_axon_model()` and `model_ota_axon_model()` in
`lib/model_ota/cmake/model_ota_axon.cmake`. All three share `model_ota_add_image()` from
`lib/model_ota/cmake/model_ota_image.cmake`
(compile a model stub, link at `partition_base + 32` with `lib/model_ota/linker/model_image.ld`,
`objcopy` the `.model_image` section, patch CRC, validate, emit the addressed hex).

Contract hashing and ``image_size`` use ``MODEL_OTA_IMAGE_LINK_BASE`` (``dt_reg_addr()`` from CMake).
Wired loaders hash the same literal and cross-check it against ``MODEL_OTA_PARTITION_ADDR()``
from the slot nodelabel before loading. See ``lib/model_ota/src/model_ota_stub_macros.h``.

Edge AI Lab solutions with an Axon backend use `model_ota_edgeai_axon_model()` in
`lib/model_ota/cmake/model_ota_axon.cmake`. The compiled Axon model is partition-loaded, and the
image additionally carries the solution's `nrf_edgeai_t` parameters. The wrapper itself stays
compiled into the app; its `model.instance.p_void` and its parameters are patched at runtime by
`nrf_edgeai_load_user_model_<id>()` from `lib/model_ota/src/model_ota_edgeai_axon_wired.c`.

Per-slot app-facing constants live in generated `model_ota/slots/<target>.h` headers
(for example `NRF_AXON_MODEL_<NAME>_PACKED_OUTPUT_SIZE`). The device-wide binding table
header is `model_ota/axon_binding_table.h`; both sit under the shared build include root
`model_ota/include/`.

Raw Axon per-image build steps wire app-owned RAM via `axon_elf.py provide` from `zephyr.elf`.
By default the linked image's optional `packed_output_buf` field is NULL; pass
`ALLOCATE_PACKED_OUTPUT` to allocate app-owned storage and wire it into the image for models that
require a dedicated packing buffer.

## Pre-flight compatibility check

Before uploading a model built against a released firmware:

```bash
python3 tools/model_ota/check_model_compat.py \
  --context /path/to/model_ota_context.json \
  --image build/ww_kws/model_ota/ww/ww_model_image.bin \
  --slot ww
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
- Build wiring: `lib/model_ota/cmake/model_ota.cmake` (single include),
  `lib/model_ota/cmake/model_ota_edgeai_neuton.cmake`,
  `lib/model_ota/cmake/model_ota_axon.cmake`, `lib/model_ota/cmake/model_ota_image.cmake`,
  `lib/model_ota/src/model_ota_edgeai_neuton_wired.c`,
  `lib/model_ota/src/model_ota_axon_image_stub.c`,
  `lib/model_ota/src/model_ota_stub_macros.h`, `lib/model_ota/linker/model_image.ld`
- Axon wiring: `tools/model_ota/axon_elf.py`, `tools/model_ota/emit_context_slot.py`
- Edge AI Lab wired loaders: `lib/model_ota/src/model_ota_edgeai_neuton_wired.c`,
  `lib/model_ota/src/model_ota_edgeai_axon_wired.c`, `include/model_ota/model_ota_edgeai.h`
