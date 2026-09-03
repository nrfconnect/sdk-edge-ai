/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef _NRF_EDGEAI_GATE_H_
#define _NRF_EDGEAI_GATE_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Weak inference-guard hooks (default implementations permit all access).
 *
 * Override from the application or model-OTA library to block inference during
 * model updates.
 *
 * ``nrf_edgeai_run_inference()`` checks ``nrf_edgeai_t.is_ota_managed`` and
 * calls session begin/end only for those contexts. To be agreed who should
 * check this flag.
 *
 * ``nrf_edgeai_guard_inference_permitted()`` is declared for potential future
 * use (for example an early exit in ``feed_inputs``) but is not called by the
 * runtime today.
 */
bool nrf_edgeai_guard_inference_permitted(const nrf_edgeai_t* p_edgeai);
int nrf_edgeai_guard_inference_session_begin(const nrf_edgeai_t* p_edgeai);
void nrf_edgeai_guard_inference_session_end(const nrf_edgeai_t* p_edgeai);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_GATE_H_ */
