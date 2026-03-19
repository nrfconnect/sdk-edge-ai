# Person recognition (nRF54L, Axon)

Runs **virat_mobilenetv2** on the Axon NPU and reports whether a **person is present** in each test picture.

Uses the compiled virat_mobilenetv2 model from `sdk-edge-ai/applications/virat_mobilenetv2/outputs`. You must compile that model first (see `applications/virat_mobilenetv2/README.md`). At build time, three pictures are converted to virat input (360×640×3 int8 HWC) and embedded for inference.

## Prerequisite

Compile the virat_mobilenetv2 model so that `../virat_mobilenetv2/outputs/nrf_axon_model_virat_mobilenetv2_.h` exists.

## Test pictures

Place these in `pictures/`:

- `demo_picture.jpeg`
- `demo_2.jpeg`
- `demo_3.jpeg`

The build runs `scripts/embed_test_images.py --model virat` to generate `src/generated/test_images.h` from these files. If any file is missing, the build fails.

## Build and run

From your NCS/sdk-edge-ai workspace (e.g. `code_ncs`):

```bash
west build -b nrf54lm20dk_nrf54lm20b_cpuapp sdk-edge-ai/applications/person_recognition
west flash
```

## Output

Serial log shows for each picture:

- **demo_picture: person present: yes/no** (person cells count)
- **demo_2: person present: yes/no** (person cells count)
- **demo_3: person present: yes/no** (person cells count)

The model output is 45×80×3 (spatial × 3 classes). Class index 1 is interpreted as "person"; the app reports how many grid cells have that class as argmax.
