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
 *   PARTITION_NODELABEL and MAX_NEURONS substituted at configure time; sets MODEL_OTA_WIRED
 *   and MODEL_OTA_NEUTON_NEURONS_CAP, defines cap-sized
 *   model_neurons_cap_[], then #includes the generated nrf_edgeai_user_model.c; defines
 *   nrf_edgeai_load_user_model_<SOLUTION_ID>() (declared via model_ota_neuton.h).
 *
 * Axon OTA build helpers (ELF probe + generated metadata, see tools/model_ota/axon_elf.py):
 *   Probe: model_ota_axon_probe.c
 *   App wired stub: model_ota_axon_app_stub.c
 *   Partition image stub: model_ota_axon_image_stub.c
 *   MODEL_OTA_AXON_EDGEAI_MODEL_SRC - optional, basename of nrf_edgeai_user_model.c; set for the
 *   Axon backend of an Edge AI Lab solution so the image also carries its nrf_edgeai_t parameters
 *
 * Axon Edge AI Lab wired (model_ota_axon_edgeai_wire):
 *   Generated: ${CMAKE_CURRENT_BINARY_DIR}/model_ota_axon_edgeai_wired_<SOLUTION_ID>.c
 *   from lib/model_ota/src/model_ota_axon_edgeai_wired.c.in
 *   PARTITION_NODELABEL substituted at configure time. Sets MODEL_OTA_WIRED before #include of
 *   generated nrf_edgeai_user_model.c;
 *   defines nrf_edgeai_load_user_model_<SOLUTION_ID>() (declared via model_ota_axon_edgeai.h).
 *
 * Every stub that has a generated solution source in scope #includes model_ota_scale_select.h
 * *after* it, to derive which scaling factors the image carries (see that header).
 */

#ifndef MODEL_OTA_STUB_MACROS_H_
#define MODEL_OTA_STUB_MACROS_H_

#include <zephyr/devicetree.h>

/** Compile-time check that @p label is a zephyr,mapped-partition node. */
#define MODEL_OTA_BUILD_ASSERT_MAPPED_PARTITION(label)                                             \
	BUILD_ASSERT(DT_MAPPED_PARTITION_EXISTS(DT_NODELABEL(label)),                               \
		     STRINGIFY(label) " must use compatible = \"zephyr,mapped-partition\"")

/** Runtime check that @p addr is the memory-mapped base from devicetree. */
#define MODEL_OTA_ASSERT_MAPPED_PARTITION_ADDR(label, addr)                                        \
	__ASSERT((addr) == (const uint8_t *)DT_MAPPED_PARTITION_ADDR(DT_NODELABEL(label)),         \
		 STRINGIFY(label) " partition_addr must match devicetree")

#endif /* MODEL_OTA_STUB_MACROS_H_ */
