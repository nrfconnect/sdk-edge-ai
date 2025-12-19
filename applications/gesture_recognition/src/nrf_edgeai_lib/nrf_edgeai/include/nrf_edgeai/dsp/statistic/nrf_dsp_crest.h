/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_crest Crest Factor (Peak to RMS)
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_CREST_FUNCTIONS_H_
#define _NRF_DSP_STAT_CREST_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates Crest Factor (Peak to RMS) in a floating-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in]       p_absmax  Optional pointer to the absolute maximum(peak) in the vector 
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.tss </pre> for calculation if it exist
 *
 * @return  flt32_t  Crest Factor (Peak to RMS) in the vector
 */
flt32_t nrf_dsp_crest_f32(const flt32_t*          p_input,
                          uint16_t                num,
                          const flt32_t*          p_absmax,
                          nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculates Crest Factor (Peak to RMS) in a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples with 'stride' in input vector
 * @param[in]       stride    Vector element offset stride
 * @param[in]       p_absmax  Optional pointer to the absolute maximum(peak) in the vector 
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.tss </pre> for calculation if it exist
 *
 * @return  flt32_t  Crest Factor (Peak to RMS) in the vector
 */
flt32_t nrf_dsp_crest_f32_s(const flt32_t*          p_input,
                            uint16_t                num,
                            size32_t                stride,
                            const flt32_t*          p_absmax,
                            nrf_dsp_stat_ctx_f32_t* p_ctx);

/**
 * @brief Calculates Crest Factor (Peak to RMS) in a INT8 fixed-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in]       p_absmax  Optional pointer to the absolute maximum(peak) in the vector 
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.tss </pre> for calculation if it exist
 *
 * @return  int16_t  Crest Factor (Peak to RMS) in the vector, multiplied by NRF_DSP_FLOAT_TO_INT_FACTOR
 */
int16_t nrf_dsp_crest_i8(const int8_t*          p_input,
                         uint16_t               num,
                         const int8_t*          p_absmax,
                         nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculates Crest Factor (Peak to RMS) in a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples with 'stride' in input vector
 * @param[in]       stride    Vector element offset stride
 * @param[in]       p_absmax  Optional pointer to the absolute maximum(peak) in the vector 
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.tss </pre> for calculation if it exist
 *
 * @return  int16_t  Crest Factor (Peak to RMS) in the vector, multiplied by NRF_DSP_FLOAT_TO_INT_FACTOR
 */
int16_t nrf_dsp_crest_i8_s(const int8_t*          p_input,
                           uint16_t               num,
                           size32_t               stride,
                           const int8_t*          p_absmax,
                           nrf_dsp_stat_ctx_i8_t* p_ctx);

/**
 * @brief Calculates Crest Factor (Peak to RMS) in a INT16 fixed-point vector.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples in input vector
 * @param[in]       p_absmax  Optional pointer to the absolute maximum(peak) in the vector 
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.tss </pre> for calculation if it exist
 *
 * @return  int16_t  Crest Factor (Peak to RMS) in the vector, multiplied by NRF_DSP_FLOAT_TO_INT_FACTOR
 */
int16_t nrf_dsp_crest_i16(const int16_t*          p_input,
                          uint16_t                num,
                          const int16_t*          p_absmax,
                          nrf_dsp_stat_ctx_i16_t* p_ctx);

/**
 * @brief Calculates Crest Factor (Peak to RMS) in a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]       p_input   Pointer to the input vector
 * @param[in]       num       Number of samples with 'stride' in input vector
 * @param[in]       stride    Vector element offset stride
 * @param[in]       p_absmax Optional pointer to the absolute maximum(peak) in the vector 
 * @param[in, out]  p_ctx     Optional pointer to the statistics context,
 *                            this function will use <pre> p_ctx->value.tss </pre> for calculation if it exist
 *
 * @return  int16_t  Crest Factor (Peak to RMS) in the vector, multiplied by NRF_DSP_FLOAT_TO_INT_FACTOR
 */
int16_t nrf_dsp_crest_i16_s(const int16_t*          p_input,
                            uint16_t                num,
                            size32_t                stride,
                            const int16_t*          p_absmax,
                            nrf_dsp_stat_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_CREST_FUNCTIONS_H_ */

/**
 * @}
 */
