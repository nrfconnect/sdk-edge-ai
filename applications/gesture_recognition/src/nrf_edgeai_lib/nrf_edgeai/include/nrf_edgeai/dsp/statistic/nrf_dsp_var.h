/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_var Variance
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_VAR_FUNCTIONS_H_
#define _NRF_DSP_STAT_VAR_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Variance value of a floating-point vector.
 * @details This function uses internal accumulator of flt32_t size, 
 *          for calculating sum of squares of values, so it will lose precision if
 *          values ​​less than 10e-4
 *
 * @param[in]        p_input   Pointer to the input vector
 * @param[in]        num       Number of samples in input vector
 * @param[in, out]   p_ctx     Optional pointer to the statistics context,
 *                             this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *                             the calculated value of variance will be written to context for further use
 *
 * @return  flt32_t  Variance value of the vector
 */
flt32_t nrf_dsp_var_f32(const flt32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Variance value of a floating-point vector ​​using values ​​in increments of 'stride'.
 * @details This function uses internal accumulator of flt32_t size, 
 *          for calculating sum of squares of values, so it will lose precision if
 *          values ​​less than 10e-4
 *
 * @param[in]        p_input   Pointer to the input vector
 * @param[in]        num       Number of samples with 'stride' in input vector
 * @param[in]        stride    Vector element offset stride
 * @param[in, out]   p_ctx     Optional pointer to the statistics context,
 *                             this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *                             the calculated value of variance will be written to context for further use
 *
 * @return  flt32_t  Variance value of the vector
 */
flt32_t nrf_dsp_var_f32_s(const flt32_t*          p_input,
                          uint16_t                num,
                          size32_t                stride,
                          nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Derivatives of variance for floating-point format
 */
typedef struct nrf_dsp_derivative_var_f32_s
{
    /** First Derivative of Variance */
    flt32_t first;

    /** Second Derivative of Variance */
    flt32_t second;
} nrf_dsp_derivative_var_f32_t;

/**
 * @brief Calculate Derivatives of variance in the floating-point vector.
 * @details This function uses internal accumulator of flt32_t size, 
 *          for calculating sum of squares of values, so it will lose precision if
 *          values ​​less than 10e-4
 *
 * @param[in]   p_input            Pointer to the input vector
 * @param[in]   num                Number of samples in input vector
 * @param[out]  p_deriv_variance   Calculated derivatives of variance @ref nrf_dsp_derivative_var_f32_t
 */
void nrf_dsp_deriv_var_f32(const flt32_t*                p_input,
                           uint16_t                      num,
                           nrf_dsp_derivative_var_f32_t* p_deriv_variance);

/**
 * @brief Calculate Variance value of a INT8 fixed-point vector.
 *
 * @param[in]        p_input   Pointer to the input vector
 * @param[in]        num       Number of samples in input vector
 * @param[in, out]   p_ctx     Optional pointer to the statistics context, this function will use
 *                              <pre> p_ctx->value.sum and/or p_ctx->value.tss </pre> for calculation if it exist,
 *                             the calculated value of variance will be written to context for further use
 *
 * @return  uint16_t  Variance value of the vector
 */
uint16_t nrf_dsp_var_i8(const int8_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Variance value of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input         Pointer to the input vector
 * @param[in]       num             Number of samples with 'stride' in input vector
 * @param[in]       stride          Vector element offset stride
 * @param[in, out]   p_ctx          Optional pointer to the statistics context, this function will use
 *                                   <pre> p_ctx->value.sum and/or p_ctx->value.tss </pre> for calculation if it exist,
 *                                  the calculated value of variance will be written to context for further use
 *
 * @return  uint16_t  Variance value of the vector
 */
uint16_t nrf_dsp_var_i8_s(const int8_t*          p_input,
                          uint16_t               num,
                          size32_t               stride,
                          nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Derivatives of variance for INT8 format
 */
typedef struct nrf_dsp_derivative_var_i8_s
{
    /** First Derivative of Variance */
    uint16_t first;

    /** Second Derivative of Variance*/
    uint16_t second;
} nrf_dsp_derivative_var_i8_t;

/**
 * @brief Calculate Derivatives of variance in the INT8 fixed-point vector.
 *
 * @param[in]   p_input            Pointer to the input vector
 * @param[in]   num                Number of samples in input vector
 * @param[out]  p_deriv_variance   Calculated derivatives of variance @ref nrf_dsp_derivative_var_i8_t
 */
void nrf_dsp_deriv_var_i8(const int8_t*                p_input,
                          uint16_t                     num,
                          nrf_dsp_derivative_var_i8_t* p_deriv_variance);

/**
 * @brief Calculate Variance value of a INT16 fixed-point vector.
 *
 * @param[in]        p_input   Pointer to the input vector
 * @param[in]        num       Number of samples in input vector
 * @param[in, out]   p_ctx     Optional pointer to the statistics context, this function will use 
 *                              <pre> p_ctx->value.sum and/or p_ctx->value.tss </pre> for calculation if it exist,
 *                             the calculated value of variance will be written to context for further use
 *
 * @return  uint32_t  Variance value of the vector
 */
uint32_t nrf_dsp_var_i16(const int16_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculate Variance value of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]        p_input         Pointer to the input vector
 * @param[in]        num             Number of samples with 'stride' in input vector
 * @param[in]        stride          Vector element offset stride
 * @param[in, out]   p_ctx           Optional pointer to the statistics context, this function will use
 *                                    <pre> p_ctx->value.sum and/or p_ctx->value.tss </pre> for calculation if it exist,
 *                                   the calculated value of variance will be written to context for further use
 *
 * @return  uint32_t  Variance value of the vector
 */
uint32_t nrf_dsp_var_i16_s(const int16_t*          p_input,
                           uint16_t                num,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Derivatives of variance for INT16 format
 */
typedef struct nrf_dsp_derivative_var_i16_s
{
    /** First Derivative of Variance */
    uint32_t first;

    /** Second Derivative of Variance */
    uint32_t second;
} nrf_dsp_derivative_var_i16_t;

/**
 * @brief Calculate Derivatives of variance in the INT16 fixed-point vector.
 *
 * @param[in]   p_input            Pointer to the input vector
 * @param[in]   num                Number of samples in input vector
 * @param[out]  p_deriv_variance   Calculated derivatives of variance @ref nrf_dsp_derivative_var_i16_t
 */
void nrf_dsp_deriv_var_i16(const int16_t*                p_input,
                           uint16_t                      num,
                           nrf_dsp_derivative_var_i16_t* p_deriv_variance);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_VAR_FUNCTIONS_H_ */

/**
 * @}
 */
