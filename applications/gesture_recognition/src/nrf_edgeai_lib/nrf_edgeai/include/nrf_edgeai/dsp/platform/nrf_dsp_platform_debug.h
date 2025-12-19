/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_debug Debug Utils & Macros
 * @{
 *
 * @brief
 *
 */

#ifndef _NRF_DSP_DEBUG_H_
#define _NRF_DSP_DEBUG_H_

#include <nrf_edgeai/nrf_edgeai_macro.h>

/**
 * @brief Macro for return values ​​to indicate potential divide-by-zero in the DSP functions
 * 
 */
#ifndef NRF_DSP_ZERO_DIV_FLAG
#define NRF_DSP_ZERO_DIV_FLAG 0
#endif

/**
 * @brief Macro for verifying that the provided condition is true. It will cause the exterior
 *        function to return an error value if it is not, only if runtime checks are enabled.
 *
 * @param[in] is_true     Boolean condition to check.
 * @param[in] ret_val     The value to be returned if the checks fails
 */
#ifdef NRF_DSP_ENABLE_RUNTIME_CHECKS
#define NRF_DSP_RETURN_IF(is_true, ret_val) RETURN_IF(is_true, ret_val)
#else
#define NRF_DSP_RETURN_IF(...) ;
#endif  // NRF_DSP_ENABLE_RUNTIME_CHECKS

/**
 * @brief Logging macro, can be enabled by setting NRF_DSP_LOG_ENABLE to 1
 * 
 */
#ifndef NRF_DSP_LOG_ENABLE
#define NRF_DSP_LOG_ENABLE 0
#endif

#if NRF_DSP_LOG_ENABLE == 1
#include <stdio.h>
#define NRF_DSP_LOG(...) printf(__VA_ARGS__)
#else
#define NRF_DSP_LOG(...)
#endif  // NRF_DSP_LOG_ENABLE

/**
 * @brief Assertion macro, can be enabled by setting NRF_DSP_ASSERT_ENABLE to 1
 * 
 */
#ifndef NRF_DSP_ASSERT_ENABLE
#define NRF_DSP_ASSERT_ENABLE 0
#endif

#if NRF_DSP_ASSERT_ENABLE == 1
#include <assert.h>
#define nrf_dsp_assert(x) assert(x)
#else
#define nrf_dsp_assert(x) ;
#endif  // NRF_DSP_ASSERT_ENABLE

#endif /* _NRF_DSP_DEBUG_H_ */

/**
 * @}
 */
