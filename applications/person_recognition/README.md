# Person recognition (nRF54L, Axon)

Runs **tinyml_vww** (Visual Wake Word) on the Axon NPU and reports whether a **person is present** in each test picture.

Uses the pre-compiled tinyml_vww model from `sdk-edge-ai/tests/axon/compiled_models`. At build time, three pictures are converted to VWW input (96×96×3 int8) and embedded for inference.

## Test pictures

Place these in `pictures/`:

- `demo_picture.jpeg`
- `demo_2.jpeg`
- `demo_3.jpeg`

The build runs `scripts/embed_test_images.py` to generate `src/generated/test_images.h` from these files. If any file is missing, the build fails.

## Build and run

From your NCS/sdk-edge-ai workspace (e.g. `code_ncs`):

```bash
west build -b nrf54lm20dk_nrf54lm20b_cpuapp sdk-edge-ai/applications/person_recognition
west flash
```

## Output

Serial log shows for each picture:

- **demo_picture: person present: yes/no** (class, score)
- **demo_2: person present: yes/no** (class, score)
- **demo_3: person present: yes/no** (class, score)
