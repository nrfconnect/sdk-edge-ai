/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_absmean Absolute Mean
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_ABSMEAN_FUNCTIONS_H_
#define _NRF_DSP_STAT_ABSMEAN_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Absolute Mean value of a floating-point vector.
 *
 * @param[in]   p_input  Pointer to the input vector
 * @param[in]   num      Number of samples in input vector
 * @param[in]   p_ctx    Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  flt32_t  Absolute Mean value of the vector
 */
flt32_t nrf_dsp_absmean_f32(const flt32_t* p_input, uint16_t num, nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Absolute Mean value of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  flt32_t  Absolute Mean value of the vector
 */
flt32_t nrf_dsp_absmean_f32_s(const flt32_t*          p_input,
                              uint16_t                num,
                              size32_t                stride,
                              nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Absolute Mean value of a INT8 fixed-point vector.
 *
 * @param[in]  p_input   Pointer to the input vector
 * @param[in]  num       Number of samples in input vector
 * @param[in]  p_ctx     Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  uint8_t  Absolute Mean value of the vector
 */
uint8_t nrf_dsp_absmean_i8(const int8_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Absolute Mean value of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]  p_input   Pointer to the input vector
 * @param[in]  num       Number of samples with 'stride' in input vector
 * @param[in]  stride    Vector element offset stride
 * @param[in]  p_ctx     Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  uint8_t  Absolute Mean value of the vector
 */
uint8_t nrf_dsp_absmean_i8_s(const int8_t*          p_input,
                             uint16_t               num,
                             size32_t               stride,
                             nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculate Absolute Mean value of a INT16 fixed-point vector.
 *
 * @param[in]  p_input   Pointer to the input vector
 * @param[in]  num       Number of samples in input vector
 * @param[in]  p_ctx     Optional pointer to the statistics context,
 *                       this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  uint16_t  Absolute Mean value of the vector
 */
uint16_t nrf_dsp_absmean_i16(const int16_t* p_input, uint16_t num, nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculate Absolute Mean value of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   p_ctx     Optional pointer to the statistics context,
 *                        this function will use <pre> p_ctx->value.sum </pre> for calculation if it exist
 *
 * @return  uint16_t  Absolute Mean value of the vector
 */
uint16_t nrf_dsp_absmean_i16_s(const int16_t*          p_input,
                               uint16_t                num,
                               size32_t                stride,
                               nrf_dsp_stat_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_ABSMEAN_FUNCTIONS_H_ */

/**
 * @}
 */
