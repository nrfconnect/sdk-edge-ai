/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_moments Statistical Moments
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_MOMENTS_FUNCTIONS_H_
#define _NRF_DSP_STAT_MOMENTS_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Floating-point statistical moments container
 */
typedef union nrf_dsp_moments_f32_u
{
    struct
    {
        /** Variance */
        flt32_t var;
        /** Skewness */
        flt32_t skew;
        /** Excess Kurtosis */
        flt32_t kur;
    } moment;

    flt32_t moments[3U];
} nrf_dsp_moments_f32_t;

/**
 * @brief INT8 statistical moments container
 */
typedef union nrf_dsp_moments_i8_u
{
    struct
    {
        /** Variance */
        int16_t var;
        /** Skewness in custom integer format with high float correlation */
        int16_t skew;
        /** Kurtosis in custom integer format with high float correlation */
        int16_t kur;
    } moment;

    int16_t moments[3U];
} nrf_dsp_moments_i8_t;

/**
 * @brief INT16 statistical moments container
 */
typedef union nrf_dsp_moments_i16_u
{
    struct
    {
        /** Variance */
        int32_t var;
        /** Skewness in custom integer format with high float correlation */
        int32_t skew;
        /** Kurtosis in custom integer format with high float correlation */
        int32_t kur;
    } moment;

    int32_t moments[3U];
} nrf_dsp_moments_i16_t;

/**
 * @brief Calculate 3 higher statistical moments(Variance, Skewness, Kurtosis) of a floating-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 * @param[out]      p_m       Pointer to the moments container @ref nrf_dsp_moments_f32_t
 */
void nrf_dsp_moments_f32(const flt32_t*          p_input,
                         uint16_t                num,
                         nrf_dsp_stat_ctx_f32_t* p_ctx,
                         nrf_dsp_moments_f32_t*  p_m);

/**
 * @brief Calculate 3 higher statistical moments(Variance, Skewness, Kurtosis) of a floating-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples with 'stride' in input vector
 * @param[in]       stride    Vector element offset stride
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 * @param[out]      p_m       Pointer to the moments container @ref nrf_dsp_moments_f32_t
 */
void nrf_dsp_moments_f32_s(const flt32_t*          p_input,
                           uint16_t                num,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_f32_t* p_ctx,
                           nrf_dsp_moments_f32_t*  p_m);

/**
 * @brief Calculate 3 higher statistical moments(Variance, Skewness, Kurtosis) of a INT8 vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 * @param[out]      p_m       Pointer to the moments container @ref nrf_dsp_moments_i8_t
 */
void nrf_dsp_moments_i8(const int8_t*          p_input,
                        uint16_t               num,
                        nrf_dsp_stat_ctx_i8_t* p_ctx,
                        nrf_dsp_moments_i8_t*  p_m);

/**
 * @brief Calculate 3 higher statistical moments(Variance, Skewness, Kurtosis) of a INT16 vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 * @param[out]      p_m       Pointer to the moments container @ref nrf_dsp_moments_i16_t
 */
void nrf_dsp_moments_i16(const int16_t*          p_input,
                         uint16_t                num,
                         nrf_dsp_stat_ctx_i16_t* p_ctx,
                         nrf_dsp_moments_i16_t*  p_m);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_MOMENTS_FUNCTIONS_H_ */

/**
 * @}
 */
