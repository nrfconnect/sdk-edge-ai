/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_madf Mean Absolute Difference
 * @{
 * @ingroup nrf_dsp_statistic
 *
 *
 */
#ifndef _NRF_DSP_STAT_MEAN_ABS_DIFF_H_
#define _NRF_DSP_STAT_MEAN_ABS_DIFF_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates Mean Absolute Difference value of a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  flt32_t  Mean Absolute Difference value in the vector
 */
flt32_t nrf_dsp_madf_f32(const flt32_t* p_input, uint16_t num);

/**
 * @brief Calculates Mean Absolute Difference value of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  flt32_t  Mean Absolute Difference value in the vector
 */
flt32_t nrf_dsp_madf_f32_s(const flt32_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Calculates Mean Absolute Difference value of a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Mean Absolute Difference value in the vector
 */
int16_t nrf_dsp_madf_i8(const int8_t* p_input, uint16_t num);

/**
 * @brief Calculates Mean Absolute Difference value of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Mean Absolute Difference value in the vector
 */
int16_t nrf_dsp_madf_i8_s(const int8_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Calculates Mean Absolute Difference value of a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Mean Absolute Difference value in the vector
 */
int16_t nrf_dsp_madf_i16(const int16_t* p_input, uint16_t num);

/**
 * @brief Calculates Mean Absolute Difference value of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Mean Absolute Difference value in the vector
 */
int16_t nrf_dsp_madf_i16_s(const int16_t* p_input, uint16_t num, size32_t stride);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_MEAN_ABS_DIFF_H_ */

/**
 * @}
 */
