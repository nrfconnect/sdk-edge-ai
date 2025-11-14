/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_INPUT_SETUP_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_INPUT_SETUP_H_

#include <nrf_edgeai/rt/nrf_edgeai_input_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Setup input context for models without windowing.
 *
 * Initializes the input context for models that do not use any windowing strategy.
 * This setup is suitable for models that process single input vectors without temporal context.
 *
 * @param[in,out] p_input_ctx Pointer to the input context structure (@ref nrf_edgeai_input_t).
 * 
 * @return NRF Edge AI operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_input_setup_no_window(nrf_edgeai_input_t* p_input_ctx);

/**
 * @brief Setup input context for discrete windowing.
 *
 * Initializes the input context for models that use discrete windowing, where each window is processed independently.
 *
 * @param[in,out] p_input_ctx Pointer to the input context structure (@ref nrf_edgeai_input_t).
 * 
 * @return NRF Edge AI operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_input_setup_discrete_window(nrf_edgeai_input_t* p_input_ctx);

/**
 * @brief Setup input context for sliding windowing.
 *
 * Initializes the input context for models that use a sliding window strategy, maintaining a moving window of recent input values.
 *
 * @param[in,out] p_input_ctx Pointer to the input context structure (@ref nrf_edgeai_input_t).
 * 
 * @return NRF Edge AI operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_input_setup_sliding_window(nrf_edgeai_input_t* p_input_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_INPUT_SETUP_H_ */