/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_hjorth Hjorth Parameters
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_HJORTH_FUNCTIONS_H_
#define _NRF_DSP_STAT_HJORTH_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Hjorth Parameters values for F32 signal format
 * 
 */
typedef struct nrf_dsp_hjorth_params_f32_s
{
    /** Hjorth Activity (signal Variance) */
    flt32_t activity;

    /** Hjorth Mobility parameter represents the mean frequency or the proportion of standard deviation 
     * of the power spectrum. This is defined as the square root of variance of the first derivative of the signal y(t)
     * divided by variance of the signal y(t). */
    flt32_t mobility;

    /** Hjorth Complexity parameter represents the change in frequency. 
     * The parameter compares the signal's similarity to a pure sine wave, 
     * where the value converges to 1 if the signal is more similar.
     */
    flt32_t complexity;
} nrf_dsp_hjorth_params_f32_t;

/**
 * @brief Hjorth Parameters values for INT8 signal format
 * 
 */
typedef struct nrf_dsp_hjorth_params_i8_s
{
    /** Hjort Activity (signal Variance) */
    int16_t activity;

    /** Hjort Mobility, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR */
    int16_t mobility;

    /** Hjort Complexity, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR */
    int16_t complexity;
} nrf_dsp_hjorth_params_i8_t;

/**
 * @brief Hjorth Parameters values for INT16 signal format
 * 
 */
typedef struct nrf_dsp_hjorth_params_i16_s
{
    /** Hjort Activity (signal Variance) */
    int32_t activity;

    /** Hjort Mobility, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR */
    int32_t mobility;

    /** Hjort Complexity, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR */
    int32_t complexity;
} nrf_dsp_hjorth_params_i16_t;

/**
 * @brief Calculates Hjorth Parameters in a floating-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[out]      p_params  Calculated Hjorth Parameters @ref nrf_dsp_hjorth_params_f32_t
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 */
void nrf_dsp_hjorth_f32(const flt32_t*               p_input,
                        uint16_t                     num,
                        nrf_dsp_hjorth_params_f32_t* p_params,
                        nrf_dsp_stat_ctx_f32_t*      p_ctx);

/**
 * @brief Calculates Hjorth Parameters in a INT8 fixed-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[out]      p_params  Calculated Hjorth Parameters @ref nrf_dsp_hjorth_params_i8_t
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 */
void nrf_dsp_hjorth_i8(const int8_t*               p_input,
                       uint16_t                    num,
                       nrf_dsp_hjorth_params_i8_t* p_params,
                       nrf_dsp_stat_ctx_i8_t*      p_ctx);

/**
 * @brief Calculates Hjorth Parameters in a INT16 fixed-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[out]      p_params  Calculated Hjorth Parameters @ref nrf_dsp_hjorth_params_i16_t
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.var </pre> for calculation if it exist
 */
void nrf_dsp_hjorth_i16(const int16_t*               p_input,
                        uint16_t                     num,
                        nrf_dsp_hjorth_params_i16_t* p_params,
                        nrf_dsp_stat_ctx_i16_t*      p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_HJORTH_FUNCTIONS_H_ */

/**
 * @}
 */
