/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_lrp Linear Regression Parameters
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_LRP_FUNCTIONS_H_
#define _NRF_DSP_STAT_LRP_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct nrf_dsp_linear_reg_params_f32_s
{
    /* Slope of linear regression */
    flt32_t slope;

    /* Intercept of linear regression */
    flt32_t intercept;
} nrf_dsp_linear_reg_params_f32_t;

typedef struct nrf_dsp_linear_reg_params_i16_s
{
    /* Slope of linear regression */
    int32_t slope;

    /* Intercept of linear regression */
    int32_t intercept;
} nrf_dsp_linear_reg_params_i16_t;

typedef struct nrf_dsp_linear_reg_params_i8_s
{
    /* Slope of linear regression */
    int16_t slope;

    /* Intercept of linear regression */
    int16_t intercept;
} nrf_dsp_linear_reg_params_i8_t;

/**
 * @brief Calculate Linear Regression Parameters (Slope, Intercept) of a floating-point vector.
 *
 * @param[in]   p_input  Pointer to the input vector
 * @param[in]   num      Number of samples in input vector
 * @param[out]  p_params Calculated Linear Regression Parameters (Slope, Intercept), @ref nrf_dsp_linear_reg_params_f32_t
 * @param[in]   p_ctx    Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 */
void nrf_dsp_lrp_f32(const flt32_t*                   p_input,
                     uint16_t                         num,
                     nrf_dsp_linear_reg_params_f32_t* p_params,
                     nrf_dsp_stat_ctx_f32_t*          p_ctx);

/**
 * @brief Calculate Linear Regression Parameters (Slope, Intercept) of a INT16 fixed-point vector.
 *
 * @param[in]  p_input   Pointer to the input vector
 * @param[in]  num       Number of samples in input vector
 * @param[out] p_params  Calculated Linear Regression Parameters (Slope, Intercept), @ref nrf_dsp_linear_reg_params_i16_t 
 * @param[in]  p_ctx     Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 */
void nrf_dsp_lrp_i16(const int16_t*                   p_input,
                     uint16_t                         num,
                     nrf_dsp_linear_reg_params_i16_t* p_params,
                     nrf_dsp_stat_ctx_i16_t*          p_ctx);

void nrf_dsp_lrp_i8(const int8_t*                   p_input,
                    uint16_t                        num,
                    nrf_dsp_linear_reg_params_i8_t* p_params,
                    nrf_dsp_stat_ctx_i8_t*          p_ctx);
#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_LRP_FUNCTIONS_H_ */

/**
 * @}
 */
