/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_clip Clipping
 * @{
 * @ingroup nrf_dsp_support
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SUPPORT_CLIPPING_H_
#define _NRF_DSP_SUPPORT_CLIPPING_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Clips floating-point values to the limit (min, max)
 * 
 * @param[in, out] p_input  Pointer to the input vector
 * @param[in]      num      Number of samples in the input vector
 * @param[in]      min      Minimum limit for the input samples, all value below will be clipped to this limit
 * @param[in]      max      Maximum limit for the input samples, all value above will be clipped to this limit
 */
void nrf_dsp_clip_f32(flt32_t* p_input, uint16_t num, const flt32_t min, const flt32_t max);

/**
 * @brief Clips INT8 values to the limit (min, max)
 * 
 * @param[in, out] p_input  Pointer to the input vector
 * @param[in]      num      Number of samples in the input vector
 * @param[in]      min      Minimum limit for the input samples, all value below will be clipped to this limit
 * @param[in]      max      Maximum limit for the input samples, all value above will be clipped to this limit
 */
void nrf_dsp_clip_i8(int8_t* p_input, uint16_t num, const int8_t min, const int8_t max);

/**
 * @brief Clips INT16 values to the limit (min, max)
 * 
 * @param[in, out] p_input  Pointer to the input vector
 * @param[in]      num      Number of samples in the input vector
 * @param[in]      min      Minimum limit for the input samples, all value below will be clipped to this limit
 * @param[in]      max      Maximum limit for the input samples, all value above will be clipped to this limit
 */
void nrf_dsp_clip_i16(int16_t* p_input, uint16_t num, const int16_t min, const int16_t max);

/** @def nrf_dsp_clip
 *
 * @brief Macro for clipping variable to the limit (min, max). 
 * 
 * @param[in] input    Variable for clipping
 * @param[in] min      Minimum limit for the variable, value below will be clipped to this limit
 * @param[in] max      Maximum limit for the variable, value above will be clipped to this limit
 * 
 * @return Clipped variable
 */
#define nrf_dsp_clip(input, min, max) \
    (((input) < (min)) ? (min) : (((input) > (max)) ? (max) : (input)))

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SUPPORT_CLIPPING_H_ */

/**
 * @}
 */
