/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_range Range
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_RANGE_FUNCTIONS_H_
#define _NRF_DSP_STAT_RANGE_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the floating point vector.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_f32().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *  
 * @return flt32_t Difference between the minimum and maximum value in the input vector
 */
flt32_t nrf_dsp_range_f32(const flt32_t* p_input, uint16_t num);

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the floating point vector using values in increments of 'stride'.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_f32_s().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   stride    Vector element offset stride
 *  
 * @return flt32_t Difference between the minimum and maximum value in the input vector
 */
flt32_t nrf_dsp_range_f32_s(const flt32_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the INT8 fixed-point vector.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_i8().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *  
 * @return uint8_t Difference between the minimum and maximum value in the input vector
 */
uint8_t nrf_dsp_range_i8(const int8_t* p_input, uint16_t num);

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the INT8 fixed-point vector using values in increments of 'stride'.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_i8_s().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   stride    Vector element offset stride
 *  
 * @return uint8_t Difference between the minimum and maximum value in the input vector
 */
uint8_t nrf_dsp_range_i8_s(const int8_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the INT16 fixed-point vector.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_i16().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *  
 * @return uint16_t Difference between the minimum and maximum value in the input vector
 */
uint16_t nrf_dsp_range_i16(const int16_t* p_input, uint16_t num);

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the INT16 fixed-point vector using values in increments of 'stride'.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_i16_s().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   stride    Vector element offset stride
 *  
 * @return uint16_t Difference between the minimum and maximum value in the input vector
 */
uint16_t nrf_dsp_range_i16_s(const int16_t* p_input, uint16_t num, size32_t stride);

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the INT32 fixed-point vector.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_i32().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *  
 * @return uint32_t Difference between the minimum and maximum value in the input vector
 */
uint32_t nrf_dsp_range_i32(const int32_t* p_input, uint16_t num);

/**
 * @brief Finds the difference between the minimum and maximum value
 *        in the INT32 fixed-point vector using values in increments of 'stride'.
 * @details This function finds min/max values via a call to @ref nrf_dsp_min_max_i32_s().
 * 
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   stride    Vector element offset stride
 *  
 * @return uint32_t Difference between the minimum and maximum value in the input vector
 */
uint32_t nrf_dsp_range_i32_s(const int32_t* p_input, uint16_t num, size32_t stride);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_RANGE_FUNCTIONS_H_ */

/**
 * @}
 */
