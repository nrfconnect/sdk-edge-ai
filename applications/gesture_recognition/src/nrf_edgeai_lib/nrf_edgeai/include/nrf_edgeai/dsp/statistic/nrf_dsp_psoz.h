/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_psot Percentage of signal over zero
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_PSOZ_FUNCTIONS_H_
#define _NRF_DSP_STAT_PSOZ_FUNCTIONS_H_

#include "nrf_dsp_psot.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculates Percentage of signal over zero in a floating-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  flt32_t  Percentage of signal over zero in the vector
 */
#define nrf_dsp_psoz_f32(p_input, num) nrf_dsp_psot_f32((p_input), (num), 0.f)

/**
 * @brief Calculates Percentage of signal over zero in a floating-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  flt32_t  Percentage of signal over zero in the vector
 */
#define nrf_dsp_psoz_f32_s(p_input, num, stride) nrf_dsp_psot_f32_s((p_input), (num), stride, 0.f)

/**
 * @brief Calculates Percentage of signal over zero in a INT8 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Percentage of signal over zero in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
#define nrf_dsp_psoz_i8(p_input, num) nrf_dsp_psot_i8((p_input), (num), 0)

/**
 * @brief Calculates Percentage of signal over zero in a INT8 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Percentage of signal over zero in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
#define nrf_dsp_psoz_i8_s(p_input, num, stride) nrf_dsp_psot_i8_s((p_input), (num), stride, 0)

/**
 * @brief Calculates Percentage of signal over zero in a INT16 fixed-point vector.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples in input vector
 *
 * @return  int16_t  Percentage of signal over zero in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
#define nrf_dsp_psoz_i16(p_input, num) nrf_dsp_psot_i16((p_input), (num), 0)

/**
 * @brief Calculates Percentage of signal over zero in a INT16 fixed-point vector ​​using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input   Pointer to the input vector
 * @param[in]   num       Number of samples with 'stride' in input vector
 * @param[in]   stride    Vector element offset stride
 *
 * @return  int16_t  Percentage of signal over zero in the vector, multiplied by NRF_DSP_PERCENTAGE_TO_INT_FACTOR
 */
#define nrf_dsp_psoz_i16_s(p_input, num, stride) nrf_dsp_psot_i16_s((p_input), (num), stride, 0)

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_PSOZ_FUNCTIONS_H_ */

/**
 * @}
 */
