/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_NN_debug Debug Utils & Macros
 * @{
 *
 * @brief
 *
 */

#ifndef _NRF_NN_DEBUG_H_
#define _NRF_NN_DEBUG_H_

#include <nrf_edgeai/nrf_edgeai_macro.h>

/**
 * @brief Macro for verifying that the provided condition is true. It will cause the exterior
 *        function to return an error value if it is not, only if runtime checks are enabled.
 *
 * @param[in] is_true     Boolean condition to check.
 * @param[in] ret_val     The value to be returned if the checks fails
 */
#ifdef NRF_NN_ENABLE_RUNTIME_CHECKS
#define NRF_NN_RETURN_IF(is_true, ret_val) RETURN_IF(is_true, ret_val)
#else
#define NRF_NN_RETURN_IF(...) ;
#endif  // NRF_NN_ENABLE_RUNTIME_CHECKS

/**
 * @brief Logging macro, can be enabled by setting NRF_NN_LOG_ENABLE to 1
 * 
 */
#ifndef NRF_NN_LOG_ENABLE
#define NRF_NN_LOG_ENABLE 0
#endif

#if NRF_NN_LOG_ENABLE == 1
#include <stdio.h>
#define NRF_NN_LOG(...) printf(__VA_ARGS__)
#else
#define NRF_NN_LOG(...)
#endif  // NRF_NN_LOG_ENABLE

/**
 * @brief Assertion macro, can be enabled by setting NRF_NN_ASSERT_ENABLE to 1
 * 
 */
#ifndef NRF_NN_ASSERT_ENABLE
#define NRF_NN_ASSERT_ENABLE 0
#endif

#if NRF_NN_ASSERT_ENABLE == 1
#include <assert.h>
#define nrf_nn_assert(x) assert(x)
#else
#define nrf_nn_assert(x) ;
#endif  // NRF_NN_ASSERT_ENABLE

#endif /* _NRF_NN_DEBUG_H_ */

/**
 * @}
 */
