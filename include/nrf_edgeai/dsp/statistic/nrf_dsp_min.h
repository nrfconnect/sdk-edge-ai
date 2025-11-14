/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_min Minimum
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_MIN_FUNCTIONS_H_
#define _NRF_DSP_STAT_MIN_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Finds Minimum value of a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  flt32_t  Minimum value in the vector
 */
flt32_t nrf_dsp_min_f32(const flt32_t* p_input, uint16_t num);

/**
 * @brief Finds Minimum value of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  flt32_t  Minimum value in the vector
 */
flt32_t nrf_dsp_min_f32_s(const flt32_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Finds Minimum value of a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int8_t  Minimum value in the vector
 */
int8_t nrf_dsp_min_i8(const int8_t* p_input, uint16_t num);

/**
 * @brief Finds Minimum value of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int8_t  Minimum value in the vector
 */
int8_t nrf_dsp_min_i8_s(const int8_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Finds Minimum value of a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Minimum value in the vector
 */
int16_t nrf_dsp_min_i16(const int16_t* p_input, uint16_t num);

/**
 * @brief Finds Minimum value of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Minimum value in the vector
 */
int16_t nrf_dsp_min_i16_s(const int16_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Finds Minimum value of a INT32 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int32_t  Minimum value in the vector
 */
int32_t nrf_dsp_min_i32(const int32_t* p_input, uint16_t num);

/**
 * @brief Finds Minimum value of a INT32 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int32_t  Minimum value in the vector
 */
int32_t nrf_dsp_min_i32_s(const int32_t* p_input, uint16_t num, size32_t stride);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_MIN_FUNCTIONS_H_ */

/**
 * @}
 */
