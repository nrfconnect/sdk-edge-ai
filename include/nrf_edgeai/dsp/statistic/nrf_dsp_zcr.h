/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_zcr Zero-crossing Rate
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_ZCR_FUNCTIONS_H_
#define _NRF_DSP_STAT_ZCR_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Counts Zero-crossing rate of a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  flt32_t  Zero-crossing rate in the vector
 */
flt32_t nrf_dsp_zcr_f32(const flt32_t* p_input, uint16_t num);

/**
 * @brief Counts Zero-crossing rate of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  flt32_t  Zero-crossing rate in the vector
 */
flt32_t nrf_dsp_zcr_f32_s(const flt32_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Counts Zero-crossing rate of a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Zero-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_zcr_i8(const int8_t* p_input, uint16_t num);

/**
 * @brief Counts Zero-crossing rate of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Zero-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_zcr_i8_s(const int8_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Counts Zero-crossing rate of a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Zero-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_zcr_i16(const int16_t* p_input, uint16_t num);

/**
 * @brief Counts Zero-crossing rate of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Zero-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_zcr_i16_s(const int16_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Counts Zero-crossing rate of a INT32 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Zero-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_zcr_i32(const int32_t* p_input, uint16_t num);

/**
 * @brief Counts Zero-crossing rate of a INT32 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Zero-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_zcr_i32_s(const int32_t* p_input, uint16_t num, size32_t stride);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_ZCR_FUNCTIONS_H_ */

/**
 * @}
 */
