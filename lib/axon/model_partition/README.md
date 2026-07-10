# Axon updatable model partition (POC)

Proof-of-concept for placing Axon compiled models in a dedicated flash partition
instead of linking them into the application image. The goal is to support future
over-the-air model updates without reflashing the full application.

Samples and applications opt in via Kconfig (`CONFIG_HELLO_AXON_MODEL_IN_PARTITION`,
`CONFIG_PERSON_DET_MODEL_IN_PARTITION`) and a devicetree `axon_model_partition`
node (`compatible = "zephyr,mapped-partition"`).

## Design overview (version 5)

Earlier iterations generated a large `*_model_image.c` file that duplicated every
weight and constant from the compiler-generated model header. Version 5 replaces
that with a **fixed stub** plus a **small generated fixups header**:

```
compiler model header (nrf_axon_model_*.h)
        │
        ├─► application link  ──► zephyr.elf (no model rodata)
        │
        └─► model image link  ──► *_model_image.bin
                ▲
                │  stub C + fixups header + app symbol addresses
```

### Build pipeline

1. **Application link** — The app is built without the model header. Symbols that
   the model references at runtime (`nrf_axon_interlayer_buffer`, packed output
   buffers, op-extension tables) live only in the application ELF.

2. **Symbol extraction** — `extract_elf_syms.py` reads those symbol addresses
   from `zephyr.elf` and emits a companion linker script with `PROVIDE()` entries.
   Thumb function pointers for `nrf_axon_nn_op_extension_*` get the LSB set.

3. **Fixups header** — `gen_axon_model_partition_fixups.py` parses the model
   header and generates a small `*_model_fixups.h` that:
   - disables `NRF_AXON_MODEL_ALLOCATE_PACKED_OUTPUT_BUFFER` (buffers are app-owned),
   - nulls out packed-output buffer macros,
   - `#include`s the original model header so rodata is compiled once.

4. **Model image link** — `model_image_stub.c` is compiled with both headers
   (`-include fixups.h`, `-include syms.h`) and linked at the partition base
   address using `linker/model_image.ld`. The stub places a partition header
   (`nrf_axon_model_image_partition_hdr`) before the model rodata.

5. **Validation** — `validate_model_partition_layout.py` checks magic, version,
   header fields, linker anchors, and that `model_offset` points at the compiled
   model symbol.

6. **Hex merge** — The partition binary is converted to Intel HEX at the
   partition offset and merged into `zephyr.hex` via `mergehex.py`.

7. **Usage report** — `report_model_partition_usage.py` prints a linker-style
   table showing how much of the devicetree partition size the image consumes.

### Runtime access

Applications load the model with:

```c
model = nrf_axon_model_partition_get(PARTITION_ADDRESS(axon_model_partition));
```

`nrf_axon_model_partition_get()` validates the header magic/version and returns a
pointer to `nrf_axon_nn_compiled_model_s` inside the mapped partition. Pointer
fields in the model struct were resolved to absolute flash/RAM addresses during
the model-image link.

### Partition image layout

```
┌──────────────────────────────────────┐  partition base (DT reg offset)
│ nrf_axon_model_partition_header      │  magic, version, model_offset, image_size
├──────────────────────────────────────┤
│ model rodata (.rodata from header)   │  weights, cmd buffers, compiled model struct
└──────────────────────────────────────┘
```

`NRF_AXON_MODEL_PARTITION_VERSION` is currently **5**. Bump it when the on-flash
layout or header semantics change.

### Integrating a new application

1. Add an `axon_model_partition` node to the board overlay.
2. Enable `CONFIG_NRF_AXON_MODEL_PARTITION` and the app-specific partition Kconfig.
3. In `CMakeLists.txt`, call `nrf_axon_model_partition_image()` with the generated
   model header path and partition nodelabel.
4. In `main.c`, use `nrf_axon_model_partition_get()` instead of `&model_*`.

### Limitations (POC)

- Model image is still produced at build time and merged into the flash hex; there
  is no runtime loader or signature check yet.
- App-owned symbols must exist in `zephyr.elf` before the model image can link.
- Partition size must be declared in devicetree and be large enough for the image.
