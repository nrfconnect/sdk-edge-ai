<!-- Copyright (c) 2026 Nordic Semiconductor ASA -->
<!-- SPDX-License-Identifier: LicenseRef-Nordic-5-Clause -->

# Neuton model-only OTA host tools

## Summary

A Neuton model is shipped as a self-contained, **linked partition image**: the compiled
`nrf_edgeai_model_neuton_t` descriptor and all its data are linked *at* the model partition's
flash base, with a header (see `include/model_ota/model_image.h`) holding a DIRECT pointer to the
descriptor. The device loads it with `model_image_load_neuton()` and only patches the RAM
neuron-scratch pointer; everything else is used in place from XIP flash.

Almost all of the image is produced by the compiler/linker. These two host scripts do the only
work that cannot be expressed in the toolchain:

- `patch_image_crc.py` - computes CRC-32/IEEE over the finished image binary (with the header's
  `crc32` field held at 0) and writes it back. These 4 bytes are the only host-written bytes in
  the image; the loader recomputes the CRC exactly the same way.
- `validate_model_image_layout.py` - a post-link check that fails the build if the on-flash
  header disagrees with the link: image linked at the partition base, header first, correct
  magic/format-version, `image_size == linker extent == binary size`, header `model` pointer ==
  `&model_instance_` and inside the image, and `crc32 != 0`.

## In the build

Both scripts are wired into the OTA build; there is no manual step. Building the `multi_model`
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
app-image payload discard lives in `lib/model_ota/cmake/model_ota_neuton.cmake`.

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
  `lib/model_ota/cmake/model_ota_neuton.cmake`, `lib/model_ota/src/model_image_stub_body.h`,
  `lib/model_ota/linker/model_image.ld`
