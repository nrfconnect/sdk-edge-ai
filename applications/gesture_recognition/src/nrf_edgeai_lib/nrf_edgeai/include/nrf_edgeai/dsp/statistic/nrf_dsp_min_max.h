/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_min_max Minimum and Maximum
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_MIN_MAX_FUNCTIONS_H_
#define _NRF_DSP_STAT_MIN_MAX_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Finds Maximum and Minimum value of a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_min     Pointer to store Minimum value in the vector
 * @param[out]  p_max     Pointer to store Maximum value in the vector
 *
 */
void nrf_dsp_min_max_f32(const flt32_t* p_input, uint16_t num, flt32_t* p_min, flt32_t* p_max);

/**
 * @brief Finds Maximum and Minimum value of a floating-point vector using values 
 *          ​​in increments of 'stride'.
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples with 'stride' in input vector
 * @param[in]  stride       Vector element offset stride
 * @param[out] p_min        Pointer to store Minimum value in the vector 
 * @param[out] p_max        Pointer to store Maximum value in the vector 
 */
void nrf_dsp_min_max_f32_s(const flt32_t* p_input,
                           uint16_t       num,
                           size32_t       stride,
                           flt32_t*       p_min,
                           flt32_t*       p_max);

/**
 * @brief Finds Maximum and Minimum value of INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_min     Pointer to store Minimum value in the vector
 * @param[out]  p_max     Pointer to store Maximum value in the vector
 *
 */
void nrf_dsp_min_max_i8(const int8_t* p_input, uint16_t num, int8_t* p_min, int8_t* p_max);

/**
 * @brief Finds Maximum and Minimum value of a INT8 fixed-point vector using values 
 *          ​​in increments of 'stride'.
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples with 'stride' in input vector
 * @param[in]  stride       Vector element offset stride
 * @param[out] p_min        Pointer to store Minimum value in the vector 
 * @param[out] p_max        Pointer to store Maximum value in the vector 
 */
void nrf_dsp_min_max_i8_s(const int8_t* p_input,
                          uint16_t      num,
                          size32_t      stride,
                          int8_t*       p_min,
                          int8_t*       p_max);

/**
 * @brief Finds Maximum and Minimum value of INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_min     Pointer to store Minimum value in the vector
 * @param[out]  p_max     Pointer to store Maximum value in the vector
 *
 */
void nrf_dsp_min_max_i16(const int16_t* p_input, uint16_t num, int16_t* p_min, int16_t* p_max);

/**
 * @brief Finds Maximum and Minimum value of a INT16 fixed-point vector using values 
 *          ​​in increments of 'stride'.
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples with 'stride' in input vector
 * @param[in]  stride       Vector element offset stride
 * @param[out] p_min        Pointer to store Minimum value in the vector 
 * @param[out] p_max        Pointer to store Maximum value in the vector 
 */
void nrf_dsp_min_max_i16_s(const int16_t* p_input,
                           uint16_t       num,
                           size32_t       stride,
                           int16_t*       p_min,
                           int16_t*       p_max);

/**
 * @brief Finds Maximum and Minimum value of INT32 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[out]  p_min     Pointer to store Minimum value in the vector
 * @param[out]  p_max     Pointer to store Maximum value in the vector
 *
 */
void nrf_dsp_min_max_i32(const int32_t* p_input, uint16_t num, int32_t* p_min, int32_t* p_max);

/**
 * @brief Finds Maximum and Minimum value of a INT32 fixed-point vector using values 
 *          ​​in increments of 'stride'.
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples with 'stride' in input vector
 * @param[in]  stride       Vector element offset stride
 * @param[out] p_min        Pointer to store Minimum value in the vector 
 * @param[out] p_max        Pointer to store Maximum value in the vector 
 */
void nrf_dsp_min_max_i32_s(const int32_t* p_input,
                           uint16_t       num,
                           size32_t       stride,
                           int32_t*       p_min,
                           int32_t*       p_max);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_MIN_MAX_FUNCTIONS_H_ */

/**
 * @}
 */
