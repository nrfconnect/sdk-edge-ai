/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_psot Percentage of signal over threshold
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_PSOT_FUNCTIONS_H_
#define _NRF_DSP_STAT_PSOT_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates Percentage of signal over threshold in a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   threshold Threshold value for counting percentage of signal over this value
 *
 * @return  flt32_t  Percentage of signal over threshold in the vector
 */
flt32_t nrf_dsp_psot_f32(const flt32_t* p_input, uint16_t num, const flt32_t threshold);

/**
 * @brief Calculates Percentage of signal over threshold in a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   threshold Threshold value for counting percentage of signal over this value
 *
 * @return  flt32_t  Percentage of signal over threshold in the vector
 */
flt32_t nrf_dsp_psot_f32_s(const flt32_t* p_input,
                           uint16_t       num,
                           size32_t       stride,
                           const flt32_t  threshold);

/**
 * @brief Calculates Percentage of signal over threshold in a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   threshold Threshold value for counting percentage of signal over this value
 *
 * @return  int16_t  Percentage of signal over threshold in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psot_i8(const int8_t* p_input, uint16_t num, const int8_t threshold);

/**
 * @brief Calculates Percentage of signal over threshold in a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   threshold Threshold value for counting percentage of signal over this value
 *
 * @return  int16_t  Percentage of signal over threshold in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psot_i8_s(const int8_t* p_input,
                          uint16_t      num,
                          size32_t      stride,
                          const int8_t  threshold);

/**
 * @brief Calculates Percentage of signal over threshold in a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 * @param[in]   threshold Threshold value for counting percentage of signal over this value
 *
 * @return  int16_t  Percentage of signal over threshold in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psot_i16(const int16_t* p_input, uint16_t num, const int16_t threshold);

/**
 * @brief Calculates Percentage of signal over threshold in a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 * @param[in]   threshold Threshold value for counting percentage of signal over this value
 *
 * @return  int16_t  Percentage of signal over threshold in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
int16_t nrf_dsp_psot_i16_s(const int16_t* p_input,
                           uint16_t       num,
                           size32_t       stride,
                           const int16_t  threshold);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_PSOT_FUNCTIONS_H_ */

/**
 * @}
 */
