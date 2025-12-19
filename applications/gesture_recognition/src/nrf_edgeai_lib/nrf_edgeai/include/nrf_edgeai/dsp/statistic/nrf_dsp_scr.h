/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_scr Sigma-crossing Rate
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_SCR_FUNCTIONS_H_
#define _NRF_DSP_STAT_SCR_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Counts Sigma-crossing rate with respect to Mean, of a floating-point vector.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples in input vector
 * @param[in]       sigma_factor    Sigma multiplier for which the crossing rate will be counted
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 *
 * @return  flt32_t  Sigma-crossing rate in the vector
 */
flt32_t nrf_dsp_scr_f32(const flt32_t*          p_input,
                        uint16_t                num,
                        nrf_dsp_sigma_factor_t  sigma_factor,
                        nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Counts Sigma-crossing rate with respect to Mean, of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples with 'stride' in input vector
 * @param[in]       sigma_factor    Sigma multiplier for which the crossing rate will be counted
 * @param[in]       stride          Vector element offset stride
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 *
 * @return  flt32_t  Sigma-crossing rate in the vector within a stride
 */
flt32_t nrf_dsp_scr_f32_s(const flt32_t*          p_input,
                          uint16_t                num,
                          nrf_dsp_sigma_factor_t  sigma_factor,
                          size32_t                stride,
                          nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Counts Sigma-crossing rate with respect to Mean, of a INT8 fixed-point vector.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples in input vector
 * @param[in]       sigma_factor    Sigma multiplier for which the crossing rate will be counted
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 *
 * @return  int16_t  Sigma-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_scr_i8(const int8_t*          p_input,
                       uint16_t               num,
                       nrf_dsp_sigma_factor_t sigma_factor,
                       nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Counts Sigma-crossing rate with respect to Mean, of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples with 'stride' in input vector
 * @param[in]       sigma_factor    Sigma multiplier for which the crossing rate will be counted
 * @param[in]       stride          Vector element offset stride
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 *
 * @return  int16_t  Sigma-crossing rate in the vector within a stride, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_scr_i8_s(const int8_t*          p_input,
                         uint16_t               num,
                         nrf_dsp_sigma_factor_t sigma_factor,
                         size32_t               stride,
                         nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Counts Sigma-crossing rate with respect to Mean, of a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 *
 * @return  int16_t  Sigma-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_scr_i16(const int16_t*          p_input,
                        uint16_t                num,
                        nrf_dsp_sigma_factor_t  sigma_factor,
                        nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Counts Sigma-crossing rate with respect to Mean, of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples with 'stride' in input vector
 * @param[in]       sigma_factor    Sigma multiplier for which the crossing rate will be counted
 * @param[in]       stride          Vector element offset stride
 * @param[in, out]  p_ctx           Optional pointer to the statistics context,
 *                                  this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 *
 * @return  int16_t  Sigma-crossing rate in the vector within a stride, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_scr_i16_s(const int16_t*          p_input,
                          uint16_t                num,
                          nrf_dsp_sigma_factor_t  sigma_factor,
                          size32_t                stride,
                          nrf_dsp_stat_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_SCR_FUNCTIONS_H_ */

/**
 * @}
 */
