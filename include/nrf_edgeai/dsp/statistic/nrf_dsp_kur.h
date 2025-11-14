/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_kur Kurtosis
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_KURTOSIS_FUNCTIONS_H_
#define _NRF_DSP_STAT_KURTOSIS_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Excess Kurtosis of a floating-point vector.
 * @note Can return NRF_DSP_ZERO_DIV_FLAG if NRF_DSP_ENABLE_RUNTIME_CHECKS is defined
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 *
 * @return  flt32_t  Excess Kurtosis value of the vector
 */
flt32_t nrf_dsp_kur_f32(const flt32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Excess Kurtosis of a floating-point vector ​​using values ​​in increments of 'stride'.
 * @note Can return NRF_DSP_ZERO_DIV_FLAG if NRF_DSP_ENABLE_RUNTIME_CHECKS is defined
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples with 'stride' in input vector
 * @param[in]       stride    Vector element offset stride
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 *
 * @return  flt32_t  Excess Kurtosis value of the vector
 */
flt32_t nrf_dsp_kur_f32_s(const flt32_t*          p_input,
                          uint16_t                num,
                          size32_t                stride,
                          nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Kurtosis of a INT8 vector.
 * @note Can return NRF_DSP_ZERO_DIV_FLAG if NRF_DSP_ENABLE_RUNTIME_CHECKS is defined.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 *
 * @return  int16_t  Kurtosis value in integer format, not real value but with high correlation to floating-point kurtosis
 */
int16_t nrf_dsp_kur_i8(const int8_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Kurtosis of a INT16 vector.
 * @note Can return NRF_DSP_ZERO_DIV_FLAG if NRF_DSP_ENABLE_RUNTIME_CHECKS is defined.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 *
 * @return  int32_t  Kurtosis value in integer format, not real value but with high correlation to floating-point kurtosis
 */
int32_t nrf_dsp_kur_i16(const int16_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_KURTOSIS_FUNCTIONS_H_ */

/**
 * @}
 */
