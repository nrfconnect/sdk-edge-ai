# Models for person_recognition

Place the TFLite file here before running the Axon compiler:

- **Expected name:** `mcunetmcunet-320kb-1mb_vww.tflite`
- **Typical source:** `~/.torch/mcunetmcunet-320kb-1mb_vww.tflite`

```bash
cp ~/.torch/mcunetmcunet-320kb-1mb_vww.tflite \
   sdk-edge-ai/applications/person_recognition/models/
```

Then compile with `compiler_mcunet_vww_320kb.yaml` (see main application `README.md`).
