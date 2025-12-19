/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_support_q Quantization
 * @{
 * @ingroup nrf_dsp_support
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SUPPORT_QUANTIZATION_H_
#define _NRF_DSP_SUPPORT_QUANTIZATION_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

/**
 * @brief Quantize the elements of the floating-point vector to INT8 vector.
 * 
 * @par Scaling and Overflow Behavior:
 *      The function uses saturating arithmetic.
 *      Results outside of the allowable INT8 range [0x80 0x7F] will be saturated.
 *
 * @note
 *      In order to apply rounding, the library should be rebuilt with the NRF_DSP_USE_MATH_ROUNDING macro
 * 
 * @param[in]  p_input  Points to the floating-point input vector
 * @param[out] p_out    Points to the INT8 output vector
 * @param[in]  num      Number of samples in input and output vector
 *
 */
void nrf_dsp_quantize_f32_to_i8(const flt32_t* p_input, int8_t* p_out, uint16_t num);

/**
 * @brief Quantize the elements of the floating-point vector to INT16 vector.
 * 
 * @par Scaling and Overflow Behavior
 *      The function uses saturating arithmetic.
 *      Results outside of the allowable INT16 range [0x8000 0x7FFF] are saturated.
 *
 * @note
 *       In order to apply rounding, the library should be rebuilt with the NRF_DSP_USE_MATH_ROUNDING macro
 * 
 * @param[in]     p_input   Points to the floating-point input vector
 * @param[out]    p_out     Points to the INT16 output vector
 * @param[in]     num       Number of samples in each vector
 *
 */
void nrf_dsp_quantize_f32_to_i16(const flt32_t* p_input, int16_t* p_out, uint16_t num);

/**
 * @brief Quantize the elements of the floating-point vector to INT32 vector.
 * 
 * @par Scaling and Overflow Behavior
 *      The function uses saturating arithmetic.
 *      Results outside of the allowable INT32 range[0x80000000 0x7FFFFFFF] are saturated.
 *
 * @note
 *       In order to apply rounding, the library should be rebuilt with the NRF_DSP_USE_MATH_ROUNDING macro
 * 
 * @param[in]     p_input   Points to the floating-point input vector
 * @param[out]    p_out     Points to the INT32 output vector
 * @param[in]     num       Number of samples in each vector
 *
 */
void nrf_dsp_quantize_f32_to_i32(const flt32_t* p_input, int32_t* p_out, uint16_t num);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SUPPORT_QUANTIZATION_H_ */

/**
 * @}
 */
