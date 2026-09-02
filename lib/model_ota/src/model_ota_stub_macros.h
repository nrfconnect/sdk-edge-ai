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
 * Neuton app wired (model_ota_edgeai_neuton_model):
 *   lib/model_ota/src/model_ota_edgeai_neuton_wired.c compiled once per solution with -D
 *   MODEL_OTA_EDGEAI_SOLUTION_ID, MODEL_OTA_EDGEAI_NEUTON_MODEL_SRC, MODEL_OTA_PARTITION_NODELABEL,
 *   MODEL_OTA_NEUTON_NEURONS_CAP and NRF_MODEL_PARTITION_ADDR; defines
 *   nrf_edgeai_load_user_model_<SOLUTION_ID>().
 *
 * Axon OTA build helpers (ELF probe + generated metadata, see tools/model_ota/axon_elf.py):
 *   Probe: model_ota_axon_probe.c
 *   App storage stub: model_ota_axon_app_stub.c (OBJECT lib, one per slot)
 *   Device-wide binding table: model_ota_axon_keep_refs.S (compiled once, merged keep list)
 *   Partition image stub: model_ota_axon_image_stub.c (raw Axon, or Edge AI Lab / Axon when
 *   MODEL_OTA_EDGEAI_AXON_MODEL_SRC is set)
 *
 * Axon Edge AI Lab wired (model_ota_edgeai_axon_model):
 *   lib/model_ota/src/model_ota_edgeai_axon_wired.c includes the per-slot private
 *   model_ota_axon_model_config.h and is compiled once per solution with -D defines; defines
 *   nrf_edgeai_load_user_model_<SOLUTION_ID>() (declared via model_ota_edgeai.h).
 *
 * Raw Axon app wired (model_ota_axon_model):
 *   lib/model_ota/src/model_ota_axon_wired.c includes the per-slot
 *   private model_ota_axon_model_config.h and is compiled once per TARGET with -D
 *   MODEL_OTA_AXON_TARGET, MODEL_OTA_PARTITION_NODELABEL and NRF_MODEL_PARTITION_ADDR; defines
 *   model_ota_load_axon_<TARGET>() (declared via model_ota_axon.h).
 *
 * Contract-hash probe (every flavour, see model_ota_contract_probe() in model_ota_common.cmake):
 *   model_ota_contract_probe.c
 *   MODEL_OTA_CONTRACT_PROBE_NEUTON / _AXON / _AXON_EDGEAI - flavour selector
 *   MODEL_OTA_CONTRACT_PROBE_MODEL_SRC - basename of nrf_edgeai_user_model.c, for the two
 *   solution flavours; NRF_MODEL_PARTITION_ADDR as above
 *
 * Every stub that has a generated solution source in scope #includes model_ota_scale_select.h
 * *after* it, to derive which scaling factors the image carries (see that header).
 */

#ifndef MODEL_OTA_STUB_MACROS_H_
#define MODEL_OTA_STUB_MACROS_H_

#include <zephyr/devicetree.h>

/** Indirection so @p label can be a macro (DT_NODELABEL token-pastes its argument). */
#define MODEL_OTA_DT_NODELABEL(label) DT_NODELABEL(label)

/** Memory-mapped partition base for a devicetree nodelabel. */
#define MODEL_OTA_PARTITION_ADDR(label)                                                            \
	((const uint8_t *)DT_MAPPED_PARTITION_ADDR(MODEL_OTA_DT_NODELABEL(label)))

/** Memory-mapped partition size for a devicetree nodelabel. */
#define MODEL_OTA_PARTITION_SIZE(label) DT_REG_SIZE(MODEL_OTA_DT_NODELABEL(label))

/** Compile-time check that @p label is a zephyr,mapped-partition node. */
#define MODEL_OTA_BUILD_ASSERT_MAPPED_PARTITION(label)                                             \
	BUILD_ASSERT(DT_MAPPED_PARTITION_EXISTS(MODEL_OTA_DT_NODELABEL(label)),                      \
		     STRINGIFY(label) " must use compatible = \"zephyr,mapped-partition\"")

#endif /* MODEL_OTA_STUB_MACROS_H_ */
