/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_tcr Threshold-crossing Rate
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_TCR_FUNCTIONS_H_
#define _NRF_DSP_STAT_TCR_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Counts Threshold-crossing rate of a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  flt32_t  Threshold-crossing rate in the vector
 */
flt32_t nrf_dsp_tcr_f32(const flt32_t* p_input, uint16_t num, const flt32_t threshold);

/**
 * @brief Counts Threshold-crossing rate of a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  flt32_t  Threshold-crossing rate in the vector
 */
flt32_t nrf_dsp_tcr_f32_s(const flt32_t* p_input,
                          uint16_t       num,
                          size32_t       stride,
                          const flt32_t  threshold);

/**
 * @brief Counts Threshold-crossing rate of a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  int16_t  Threshold-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_tcr_i8(const int8_t* p_input, uint16_t num, const int8_t threshold);

/**
 * @brief Counts Threshold-crossing rate of a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  int16_t  Threshold-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_tcr_i8_s(const int8_t* p_input,
                         uint16_t      num,
                         size32_t      stride,
                         const int8_t  threshold);

/**
 * @brief Counts Threshold-crossing rate of a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  int16_t  Threshold-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_tcr_i16(const int16_t* p_input, uint16_t num, const int16_t threshold);

/**
 * @brief Counts Threshold-crossing rate of a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  int16_t  Threshold-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_tcr_i16_s(const int16_t* p_input,
                          uint16_t       num,
                          size32_t       stride,
                          const int16_t  threshold);

/**
 * @brief Counts Threshold-crossing rate of a INT32 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  int16_t  Threshold-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_tcr_i32(const int32_t* p_input, uint16_t num, const int32_t threshold);

/**
 * @brief Counts Threshold-crossing rate of a INT32 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   threshold Threshold value for counting crossing rate
 *
 * @return  int16_t  Threshold-crossing rate in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_tcr_i32_s(const int32_t* p_input,
                          uint16_t       num,
                          size32_t       stride,
                          const int32_t  threshold);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_TCR_FUNCTIONS_H_ */

/**
 * @}
 */
