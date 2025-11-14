/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_INPUT_FEED_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_INPUT_FEED_H_

#include <nrf_edgeai/rt/nrf_edgeai_input_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE
 * @brief Macro to declare input feeding interface functions for various windowing and data types.
 *
 * All declared functions have the following prototype:
 * @code
 * nrf_edgeai_err_t nrf_edgeai_input_feed_<mode>(nrf_edgeai_input_t* p_input_ctx,
 *                                             void*              p_input_values,
 *                                             uint16_t       num_values);
 * @endcode
 * where <mode> specifies the windowing/data type strategy.
 */
#define NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(interface_name)                                 \
    nrf_edgeai_err_t nrf_edgeai_input_feed_##interface_name(nrf_edgeai_input_t* p_input_ctx,    \
                                                            void*               p_input_values, \
                                                            uint16_t            num_values)

/**
 * @brief Feed input data without windowing.
 *
 * These functions feed input data directly into the input context without applying any windowing.
 * They are suitable for models that process single vectors or do not require temporal context.
 *
 * @param[in,out] p_input_ctx   Pointer to the input context structure (@ref nrf_edgeai_input_t).
 * @param[in]     p_input_values Pointer to the input values buffer.
 * @param[in]     num_values     Number of input values to feed.
 *
 * @note
 * - Use nrf_edgeai_input_feed_no_window() for basic input feeding with no internal memory buffer.
 * - Use nrf_edgeai_input_feed_no_window_extra() if internal extra memory buffer is required.
 */
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(no_window);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(no_window_extra);

/**
 * @brief Feed input data using discrete windowing.
 *
 * These functions feed input data into the input context using a discrete windowing strategy,
 * where each window is processed independently. Masked variants allow selective feature feeding.
 *
 * @param[in,out] p_input_ctx   Pointer to the input context structure (@ref nrf_edgeai_input_t).
 * @param[in]     p_input_values Pointer to the input values buffer.
 * @param[in]     num_values     Number of input values to feed.
 *
 * @note
 * - Use nrf_edgeai_input_feed_discrete_window_i8(), nrf_edgeai_input_feed_discrete_window_i16(), or
 *   nrf_edgeai_input_feed_discrete_window_f32() for different data types.
 * - Use masked variants for selective feature feeding.
 */
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(discrete_window_i8);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(discrete_window_masked_i8);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(discrete_window_i16);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(discrete_window_masked_i16);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(discrete_window_f32);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(discrete_window_masked_f32);

/**
 * @brief Feed input data using sliding windowing.
 *
 * These functions feed input data into the input context using a sliding window strategy,
 * maintaining a moving window of recent input values. Masked variants allow selective feature feeding.
 *
 * @param[in,out] p_input_ctx   Pointer to the input context structure (@ref nrf_edgeai_input_t).
 * @param[in]     p_input_values Pointer to the input values buffer.
 * @param[in]     num_values     Number of input values to feed.
 *
 * @note
 * - Use nrf_edgeai_input_feed_sliding_window_i8(), nrf_edgeai_input_feed_sliding_window_i16(), or
 *   nrf_edgeai_input_feed_sliding_window_f32() for different data types.
 * - Use masked variants for selective feature feeding.
 */
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(sliding_window_i8);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(sliding_window_masked_i8);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(sliding_window_i16);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(sliding_window_masked_i16);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(sliding_window_f32);
NRF_EDGEAI_DECLARE_INPUT_FEED_INTERFACE(sliding_window_masked_f32);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_INPUT_FEED_H_ */