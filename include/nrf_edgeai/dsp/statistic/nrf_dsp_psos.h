/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_psos Percentage of signal over Sigma
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_PSOS_FUNCTIONS_H_
#define _NRF_DSP_STAT_PSOS_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates Percentage of signal over Sigma with respect to Mean, in the floating-point vector.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples in input vector
 * @param[in]       sigma_factor    Sigma multiplier
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var and p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  flt32_t  Percentage of signal over Sigma in the vector
 */
flt32_t nrf_dsp_psos_f32(const flt32_t*          p_input,
                         uint16_t                num,
                         nrf_dsp_sigma_factor_t  sigma_factor,
                         nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculates Percentage of signal over Sigma with respect to Mean, in the floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples with 'stride' in input vector
 * @param[in]       sigma_factor    Sigma multiplier
 * @param[in]       stride          Vector element offset stride
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var and p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  flt32_t  Percentage of signal over Sigma in the vector within a stride
 */
flt32_t nrf_dsp_psos_f32_s(const flt32_t*          p_input,
                           uint16_t                num,
                           nrf_dsp_sigma_factor_t  sigma_factor,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculates Percentage of signal over Sigma with respect to Mean, in the INT8 fixed-point vector.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples in input vector
 * @param[in]       sigma_factor    Sigma multiplier
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var and p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int16_t  Percentage of signal over Sigma in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psos_i8(const int8_t*          p_input,
                        uint16_t               num,
                        nrf_dsp_sigma_factor_t sigma_factor,
                        nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculates Percentage of signal over Sigma with respect to Mean, in the INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples with 'stride' in input vector
 * @param[in]       sigma_factor    Sigma multiplier
 * @param[in]       stride          Vector element offset stride
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var and p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int16_t  Percentage of signal over Sigma in the vector within a stride, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psos_i8_s(const int8_t*          p_input,
                          uint16_t               num,
                          nrf_dsp_sigma_factor_t sigma_factor,
                          size32_t               stride,
                          nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculates Percentage of signal over Sigma with respect to Mean, in the INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.var and p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int16_t  Percentage of signal over Sigma in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psos_i16(const int16_t*          p_input,
                         uint16_t                num,
                         nrf_dsp_sigma_factor_t  sigma_factor,
                         nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculates Percentage of signal over Sigma with respect to Mean, in the INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples with 'stride' in input vector
 * @param[in]       sigma_factor    Sigma multiplier
 * @param[in]       stride          Vector element offset stride
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var and p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int16_t  Percentage of signal over Sigma in the vector within a stride, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psos_i16_s(const int16_t*          p_input,
                           uint16_t                num,
                           nrf_dsp_sigma_factor_t  sigma_factor,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_PSOS_FUNCTIONS_H_ */

/**
 * @}
 */
