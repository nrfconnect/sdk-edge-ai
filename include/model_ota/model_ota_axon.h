/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_OTA_AXON_H_
#define MODEL_OTA_MODEL_OTA_AXON_H_

/**
 * @file
 * @brief Model-only OTA helpers for raw Axon models (no nrf_edgeai_t wrapper).
 *
 * Wired loaders are built from lib/model_ota/src/model_ota_axon_wired.c with per-model -D
 * defines from model_ota_axon_model().
 */

#include <model_ota/model_image.h>

#include <drivers/axon/nrf_axon_nn_infer.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MODEL_OTA_AXON_LOAD_DECL_(target)                                                          \
	enum model_image_result model_ota_load_axon_##target(                                      \
		const nrf_axon_nn_compiled_model_s **out)

/** Declare model_ota_load_axon_<target>() from a wired static library. */
#define MODEL_OTA_AXON_LOAD_DECL(target) MODEL_OTA_AXON_LOAD_DECL_(target)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_OTA_AXON_H_ */
