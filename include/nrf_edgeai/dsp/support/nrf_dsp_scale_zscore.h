/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_scaling_zscore Z-score Scaling
 * @{
 * @ingroup nrf_dsp_support
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SUPPORT_SCALE_ZSCORE_H_
#define _NRF_DSP_SUPPORT_SCALE_ZSCORE_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Z-score scaling of the floating-point vector ( y = (x - mean) / stddev )
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples in input vector
 * @param[in]  mean         Mean scaling factor
 * @param[in]  stddev       Standard deviation scaling factor
 * @param[out] p_output     Pointer to the output scaled vector
 */
void nrf_dsp_scale_zscore_f32(const flt32_t* p_input,
                              uint16_t       num,
                              const flt32_t  mean,
                              const flt32_t  stddev,
                              flt32_t*       p_output);

/**
 * @brief Z-score scaling of the INT8 vector ( y = (x - mean) / stddev )
 * @note In order not to lose accuracy when dividing, the subtraction result (x - mean) is multiplied by NRF_DSP_INT8_QFACTOR
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples in input vector
 * @param[in]  mean         Mean scaling factor
 * @param[in]  stddev       Standard deviation scaling factor
 * @param[out] p_output     Pointer to the output scaled vector
 */
void nrf_dsp_scale_zscore_i8(const int8_t* p_input,
                             uint16_t      num,
                             const int8_t  mean,
                             const int8_t  stddev,
                             int8_t*       p_output);

/**
 * @brief Z-score scaling of the INT16 vector ( y = (x - mean) / stddev )
 * @note In order not to lose accuracy when dividing, the subtraction result (x - mean) is multiplied by NRF_DSP_INT16_QFACTOR
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples in input vector
 * @param[in]  mean         Mean scaling factor
 * @param[in]  stddev       Standard deviation scaling factor
 * @param[out] p_output     Pointer to the output scaled vector
 */
void nrf_dsp_scale_zscore_i16(const int16_t* p_input,
                              uint16_t       num,
                              const int16_t  mean,
                              const int16_t  stddev,
                              int16_t*       p_output);

/** @def nrf_dsp_scale_zscore_int_q
 *
 * @brief Macro for Z-score scaling of integer variable with quantization
 * 
 * @param[in, out] var      Variable for scaling
 * @param[in]      qfactor  Quantization factor
 * @param[in]      mean     Mean scaling factor
 * @param[in]      stddev   Standard deviation scaling factor
 */
#define nrf_dsp_scale_zscore_int_q(var, qfactor, mean, stddev) \
    (((int32_t)(var - mean) * qfactor) / stddev)

/** @def nrf_dsp_scale_zscore_float
 *
 * @brief Macro for Z-score scaling of floating-point variable
 * 
 * @param[in, out] var    Variable for scaling
 * @param[in]      mean   Mean scaling factor
 * @param[in]      stddev Standard deviation scaling factor
 */
#define nrf_dsp_scale_zscore_float(var, mean, stddev) ((flt32_t)(var - mean) / stddev)

/** @def nrf_dsp_scale_zscore_float_q
 *
 * @brief Macro for Z-score scaling of floating-point variable with quantization
 * 
 * @param[in, out] var      Variable for scaling
 * @param[in]      qfactor  Quantization factor
 * @param[in]      mean     Mean scaling factor
 * @param[in]      stddev   Standard deviation scaling factor
 */
#define nrf_dsp_scale_zscore_float_q(var, qfactor, mean, stddev) (((var - mean) * qfactor) / stddev)

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SUPPORT_SCALE_ZSCORE_H_ */

/**
 * @}
 */
