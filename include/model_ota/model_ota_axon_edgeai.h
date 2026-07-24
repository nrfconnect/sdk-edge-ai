/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef MODEL_OTA_MODEL_OTA_AXON_EDGEAI_H_
#define MODEL_OTA_MODEL_OTA_AXON_EDGEAI_H_

/**
 * @file
 * @brief Model-only OTA helpers for Edge AI Lab models with an Axon backend.
 *
 * A "Nordic EdgeAI Lab" solution exported for the Axon backend is a nrf_edgeai_t wrapper
 * (input windowing, DSP feature pipeline, decode interfaces - all app-side glue) around a
 * compiled Axon model (nrf_axon_nn_compiled_model_s - weights, cmd buffer, persistent vars).
 * Only the compiled Axon model is swappable: it is built as a self-contained partition image
 * by model_ota_axon_model() (see model_ota_axon_edgeai.cmake), exactly like a pure-Axon model
 * (e.g. person detection). The wrapper stays compiled into the app; its
 * model.instance.p_void is patched at runtime to the loaded image's model pointer.
 *
 * Wired models are built from model_ota_axon_edgeai_wired.c.in (see
 * model_ota_axon_edgeai.cmake). Generated nrf_edgeai_user_model.c stays agnostic: it honors
 * the MODEL_OTA_AXON_RUNTIME_WIRED hook below when a wired translation unit defines it before
 * #include, skipping the (otherwise unconditional) #include of the generated Axon model header
 * so its weights are never linked into the app image.
 */

#include <stdint.h>

#include <nrf_edgeai/nrf_edgeai.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Declare nrf_edgeai_load_user_model_<solution_id>() from a wired static library. */
#define MODEL_OTA_AXON_EDGEAI_LOAD_DECL(solution_id)                                          \
	nrf_edgeai_t *nrf_edgeai_load_user_model_##solution_id(uint8_t fa_id,                  \
							       const uint8_t *partition_addr)

#ifdef __cplusplus
}
#endif

#endif /* MODEL_OTA_MODEL_OTA_AXON_EDGEAI_H_ */
