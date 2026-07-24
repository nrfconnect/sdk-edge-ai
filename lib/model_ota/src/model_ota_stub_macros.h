/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Build-time macro contract for model_ota stubs.
 *
 * Each stub translation unit is compiled once per model with macros set by the matching
 * CMake helper. Undefined required macros fail at compile time with #error.
 *
 * Neuton partition image (model_ota_neuton_image):
 *   lib/model_ota/src/model_ota_neuton_image_stub.c
 *   MODEL_OTA_NEUTON_MODEL_SRC   - basename of nrf_edgeai_user_model.c
 *   NRF_MODEL_PARTITION_ADDR     - flash base from devicetree
 *   MODEL_IMAGE_NAME_STR         - optional (default: EDGEAI_LAB_SOLUTION_ID_STR)
 *   MODEL_IMAGE_VERSION_U32      - optional (default: 0x00010000)
 *
 * Neuton app wired (model_ota_neuton_wire):
 *   Generated: ${CMAKE_CURRENT_BINARY_DIR}/model_ota_neuton_wired_<SOLUTION_ID>.c
 *   from lib/model_ota/src/model_ota_neuton_wired.c.in
 *   MAX_NEURONS substituted at configure time; sets MODEL_OTA_NEUTON_RUNTIME_WIRED and
 *   MODEL_OTA_NEUTON_NEURONS_CAP before #include of generated nrf_edgeai_user_model.c;
 *   defines nrf_edgeai_load_user_model_<SOLUTION_ID>() (declared via model_ota_neuton.h).
 *
 * Axon OTA build helpers (ELF probe + generated metadata, see tools/model_ota/axon_elf.py):
 *   Probe: model_ota_axon_probe.c
 *   App wired stub: model_ota_axon_app_stub.c
 *   Partition image stub: model_ota_axon_image_stub.c
 */

#ifndef MODEL_OTA_STUB_MACROS_H_
#define MODEL_OTA_STUB_MACROS_H_

#define MODEL_OTA_STUB_XSTR(s) #s
#define MODEL_OTA_STUB_STR(s)  MODEL_OTA_STUB_XSTR(s)

#endif /* MODEL_OTA_STUB_MACROS_H_ */
