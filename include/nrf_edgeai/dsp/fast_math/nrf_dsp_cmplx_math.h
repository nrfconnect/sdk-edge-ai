/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_cmplx_math Complex math functions
 * @{
 * @ingroup nrf_dsp_fast_math
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_FASTMATH_COMPLEX_MATH_H_
#define _NRF_DSP_FASTMATH_COMPLEX_MATH_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief   Computes the magnitude of the elements of a complex data vector.
 * 
 * @param[in]   p_input          Input array and the data is stored in an interleaved fashion [xR0, xI0, xR1, xI1, ..., xRn, xIn].
 *                               The input array should has a total of <code>2*num</code> values
 * @param[out]  p_output         Output array where results should be written.
 *                               The output array should has len of <code>num</code> samples
 * @param[in]   num              The number of complex samples in the input array
 */
void nrf_dsp_cmplx_mag_f32(const flt32_t* p_input, flt32_t* p_output, uint16_t num);

/**
 * @brief   Computes the magnitude of the elements of a complex data vector in ingeter 16-bit format.
 * 
 * @param[in]   p_input          Input array and the data is stored in an interleaved fashion [xR0, xI0, xR1, xI1, ..., xRn, xIn].
 *                               The input array should has a total of <code>2*num</code> values
 * @param[out]  p_output         Output array where results should be written.
 *                               The output array should has len of <code>num</code> samples
 * @param[in]   num              The number of complex samples in the input array
 */
void nrf_dsp_cmplx_mag_i16(const int16_t* p_input, int16_t* p_output, uint16_t num);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_FASTMATH_COMPLEX_MATH_H_ */

/**
 * @}
 */
