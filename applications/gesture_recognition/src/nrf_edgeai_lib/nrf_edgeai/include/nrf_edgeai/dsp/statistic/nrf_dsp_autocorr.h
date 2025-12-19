/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_autocorr Autocorrelation
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_AUTOCORR_FUNCTIONS_H_
#define _NRF_DSP_STAT_AUTOCORR_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Autocorrelation value in the floating-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in]       lag       Number of samples use for lag of signal
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 *
 * @return  flt32_t  Autocorrelation value in the vector
 */
flt32_t nrf_dsp_autocorr_f32(const flt32_t*          p_input,
                             uint16_t                num,
                             const uint8_t           lag,
                             nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Autocorrelation value in the INT8 fixed-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in]       lag       Number of samples use for lag of signal
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 *
 * @return  int16_t  Autocorrelation value in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_autocorr_i8(const int8_t*          p_input,
                            uint16_t               num,
                            const uint8_t          lag,
                            nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Autocorrelation value in the INT16 fixed-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in]       lag       Number of samples use for lag of signal
 * @param[in, out]  p_ctx     Optional pointer to the statistics context, this function will use
 *                            <pre> p_ctx->value.sum and/or p_ctx->value.var </pre> for calculation if it exist,
 *                            the calculated value of variance will be written to context for further use
 *
 * @return  int16_t  Autocorrelation value in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_autocorr_i16(const int16_t*          p_input,
                             uint16_t                num,
                             const uint8_t           lag,
                             nrf_dsp_stat_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_AUTOCORR_FUNCTIONS_H_ */

/**
 * @}
 */
