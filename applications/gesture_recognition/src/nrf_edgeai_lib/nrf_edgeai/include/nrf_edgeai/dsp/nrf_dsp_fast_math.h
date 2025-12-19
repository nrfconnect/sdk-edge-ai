/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_fast_math Fast Math
 * @{
 * @ingroup nrf_dsp
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_FAST_MATH_FUNCTIONS_H_
#define _NRF_DSP_FAST_MATH_FUNCTIONS_H_

#include "fast_math/nrf_dsp_cmplx_math.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate square root of the provided floating point value 
 * 
 * @param[in] x             Argument for square root calculation 
 * 
 * @return flt32_t     Square root of the provided argument
 */
flt32_t nrf_dsp_sqrt_f32(const flt32_t x);

/**
 * @brief Calculate square root of the provided UINT32 value 
 * 
 * @param[in] x             Argument for square root calculation 
 * 
 * @return uint32_t     Square root of the provided argument
 */
uint32_t nrf_dsp_sqrt_u32(const uint32_t x);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_FAST_MATH_FUNCTIONS_H_ */

/**
 * @}
 */
