/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_mean Mean
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_MEAN_FUNCTIONS_H_
#define _NRF_DSP_STAT_MEAN_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Mean value of a floating-point vector.
 *
 * @param[in]   p_input  Pointer to the input vector
 * @param[in]   num      Number of samples in input vector
 * @param[in]   p_ctx    Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  flt32_t  Mean value of the vector
 */
flt32_t nrf_dsp_mean_f32(const flt32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Mean value of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  flt32_t  Mean value of the vector
 */
flt32_t nrf_dsp_mean_f32_s(const flt32_t*          p_input,
                           uint16_t                num,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Mean value of a INT8 fixed-point vector.
 *
 * @param[in]  p_input   Pointer to the input vector
 * @param[in]  num       Number of samples in input vector
 * @param[in]  p_ctx     Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int8_t  Mean value of the vector
 */
int8_t nrf_dsp_mean_i8(const int8_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Mean value of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]  p_input   Pointer to the input vector
 * @param[in]  num       Number of samples with 'stride' in input vector
 * @param[in]  stride    Vector element offset stride
 * @param[in]  p_ctx     Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int8_t  Mean value of the vector
 */
int8_t nrf_dsp_mean_i8_s(const int8_t*          p_input,
                         uint16_t               num,
                         size32_t               stride,
                         nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Mean value of a INT16 fixed-point vector.
 *
 * @param[in]  p_input   Pointer to the input vector
 * @param[in]  num       Number of samples in input vector
 * @param[in]  p_ctx     Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int16_t  Mean value of the vector
 */
int16_t nrf_dsp_mean_i16(const int16_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculate Mean value of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int16_t  Mean value of the vector
 */
int16_t nrf_dsp_mean_i16_s(const int16_t*          p_input,
                           uint16_t                num,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculate Mean value of a INT32 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist,
 *                        the calculated value of mean will be written to context for further use
 * 
 * @return  int32_t  Mean value of the vector
 */
int32_t nrf_dsp_mean_i32(const int32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i32_t* p_ctx);

/**
 * @brief Calculate Mean value of a INT32 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  int32_t  Mean value of the vector
 */
int32_t nrf_dsp_mean_i32_s(const int32_t*          p_input,
                           uint16_t                num,
                           size32_t                stride,
                           nrf_dsp_stat_ctx_i32_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_MEAN_FUNCTIONS_H_ */

/**
 * @}
 */
