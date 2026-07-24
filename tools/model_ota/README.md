<!-- Copyright (c) 2026 Nordic Semiconductor ASA -->
<!-- SPDX-License-Identifier: LicenseRef-Nordic-5-Clause -->

# Model-only OTA host tools

## Summary

A Neuton or Axon model is shipped as a self-contained, **linked partition image**. The model
descriptor and data are linked at the model partition's flash base, with a header (see
`include/model_ota/model_image.h`) holding a direct pointer to the descriptor.

Almost all of the image is produced by the compiler/linker. These host scripts perform the work
that cannot be expressed directly in the toolchain:

- `patch_image_crc.py` - computes CRC-32/IEEE over the finished image binary (with the header's
  `crc32` field held at 0) and writes it back. These 4 bytes are the only host-written bytes in
  the image; the loader recomputes the CRC exactly the same way.
- `validate_model_image_layout.py` - a post-link check that fails the build if the on-flash
  header disagrees with the link: image linked at the partition base, header first, correct
  magic/format-version, image and partition sizes, model pointers, and CRC.
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
```

The per-image build steps live in `lib/model_ota/cmake/model_ota_neuton_image.cmake`
(compile a model stub, link at the partition base with `lib/model_ota/linker/model_image.ld`,
`objcopy` the `.model_image` section, patch CRC, validate, emit the addressed hex). The
app-image payload discard lives in `lib/model_ota/cmake/model_ota_neuton.cmake`
(`configure_file` from `lib/model_ota/src/model_ota_neuton_wired.c.in`).

Axon app wiring and image generation use one `model_ota_axon_model()` declaration in
`lib/model_ota/cmake/model_ota_axon.cmake`. By default the linked image's optional
`packed_output_buf` field is NULL and no app RAM is spent on it; pass
`ALLOCATE_PACKED_OUTPUT` to allocate app-owned storage and wire it into the image for
models that require a dedicated packing buffer (the `multi_model` sample's `person_det`
declaration uses this option, so both code paths are exercised by its OTA build).

## Flashing (separate from the app)

The app (`zephyr.hex`) and each model partition are flashed independently. Program one model
image into its partition without disturbing the app or the other partitions:

```bash
nrfutil device program --firmware gear_anomaly_model_partition.hex \
  --options chip_erase_mode=ERASE_RANGES_TOUCHED_BY_FIRMWARE,reset=RESET_SYSTEM
```

## References

- Image format and loader: `include/model_ota/model_image.h`, `lib/model_ota/model_image_neuton.c`
- Build wiring: `lib/model_ota/cmake/model_ota_neuton_image.cmake`,
  `lib/model_ota/cmake/model_ota_neuton.cmake`, `lib/model_ota/src/model_ota_neuton_wired.c.in`,
  `lib/model_ota/src/model_ota_neuton_image_stub.c`,
  `lib/model_ota/src/model_ota_stub_macros.h`, `lib/model_ota/linker/model_image.ld`
- Axon wiring: `lib/model_ota/cmake/model_ota_axon.cmake`,
  `tools/model_ota/axon_elf.py`
