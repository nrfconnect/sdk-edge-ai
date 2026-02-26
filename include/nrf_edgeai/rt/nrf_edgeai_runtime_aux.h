/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_runtime_aux nRF Edge AI Lab Runtime Auxiliary API 
 * @{
 *
 * @ingroup nrf_edgeai_runtime
 *
 */

#ifndef _NRF_EDGEAI_RUNTIME_AUX_H_
#define _NRF_EDGEAI_RUNTIME_AUX_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get number of Neuton model neurons
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
uint16_t nrf_edgeai_model_neuton_neurons_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get number of Neuton model weights
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
uint16_t nrf_edgeai_model_neuton_weights_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Initialize persistent variable buffers for Axon model. 
 * Should be called at the start of each streaming session for streaming-style models, 
 * can be called separately if persistent vars need to be re-initialized without re-initializing the whole model (e.g. between streaming sessions).
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 * @return Operational status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_model_axon_init_persistent_vars(nrf_edgeai_t* p_edgeai);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_RUNTIME_AUX_H_ */

/**
 * @}
 */
