/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_sum Sum
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_SUM_FUNCTIONS_H_
#define _NRF_DSP_STAT_SUM_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Sum of values of a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  flt32_t  Sum value of the vector
 */
flt32_t nrf_dsp_sum_f32(const flt32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Sum value of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  flt32_t  Sum value of the vector
 */
flt32_t nrf_dsp_sum_f32_s(const flt32_t*          p_input,
                          uint16_t                num,
                          size32_t                stride,
                          nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Sum value of a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  int32_t  Sum value of the vector
 */
int32_t nrf_dsp_sum_i8(const int8_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Sum value of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  int32_t  Sum value of the vector
 */
int32_t nrf_dsp_sum_i8_s(const int8_t*          p_input,
                         uint16_t               num,
                         size32_t               stride,
                         nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Sum value of a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  int32_t  Sum value of the vector
 */
int32_t nrf_dsp_sum_i16(const int16_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculate Sum value of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  int32_t  Sum value of the vector
 */
int32_t nrf_dsp_sum_i16_s(const int16_t*          p_input,
                          uint16_t                num,
                          size32_t                stride,
                          nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculate Sum value of a INT32 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  int64_t  Sum value of the vector
 */
int64_t nrf_dsp_sum_i32(const int32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i32_t* p_ctx);

/**
 * @brief Calculate Sum value of a INT32 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[out]  p_ctx     Optional pointer to the statistics context,
 *                        the calculated value of sum will be written to context for further use
 *
 * @return  int64_t  Sum value of the vector
 */
int64_t nrf_dsp_sum_i32_s(const int32_t*          p_input,
                          uint16_t                num,
                          size32_t                stride,
                          nrf_dsp_stat_ctx_i32_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_SUM_FUNCTIONS_H_ */

/**
 * @}
 */
