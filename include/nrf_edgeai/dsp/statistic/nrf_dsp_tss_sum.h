/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_tss_sum_sum Total Sum of Squares and Sum
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_TSS_SUM_FUNCTIONS_H_
#define _NRF_DSP_STAT_TSS_SUM_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Total Sum of Squares and Sum of values of a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of tss and sum will be written to context for further use
 *
 */
void nrf_dsp_tss_sum_f32(const flt32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Total Sum of Squares and Sum value of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of tss and sum will be written to context for further use
 *
 */
void nrf_dsp_tss_sum_f32_s(const flt32_t*          p_input,
                           uint16_t                num,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Total Sum of Squares and Sum value of a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of tss and sum will be written to context for further use
 *
 */
void nrf_dsp_tss_sum_i8(const int8_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Total Sum of Squares and Sum value of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of tss and sum will be written to context for further use
 *
 */
void nrf_dsp_tss_sum_i8_s(const int8_t*          p_input,
                          uint16_t               num,
                          size32_t               stride,
                          nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Total Sum of Squares and Sum value of a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of tss and sum will be written to context for further use
 *
 */
void nrf_dsp_tss_sum_i16(const int16_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculate Total Sum of Squares and Sum value of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of tss and sum will be written to context for further use
 *
 */
void nrf_dsp_tss_sum_i16_s(const int16_t*          p_input,
                           uint16_t                num,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_TSS_SUM_FUNCTIONS_H_ */

/**
 * @}
 */
