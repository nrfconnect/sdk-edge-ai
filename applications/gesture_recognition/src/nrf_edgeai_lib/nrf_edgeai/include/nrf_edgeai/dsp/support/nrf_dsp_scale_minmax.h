/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_scale_minmax Min-Max Scaling
 * @{
 * @ingroup nrf_dsp_support
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SUPPORT_SCALE_MINMAX_H_
#define _NRF_DSP_SUPPORT_SCALE_MINMAX_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Min-Max scaling of the floating-point vector ( y = (x - min) / (max - min) )
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples in input vector
 * @param[in]  min          Minimum scaling factor
 * @param[in]  max          Maximum scaling factor
 * @param[out] p_output     Pointer to the output scaled vector
 */
void nrf_dsp_scale_minmax_f32(const flt32_t* p_input,
                              uint16_t       num,
                              const flt32_t  min,
                              const flt32_t  max,
                              flt32_t*       p_output);

/**
 * @brief Min-Max scaling of the INT8 vector ( y = (x - min) / (max - min) )
 * @note In order not to lose accuracy when dividing, the subtraction result (x - min) is multiplied by NRF_DSP_INT8_QFACTOR
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples in input vector
 * @param[in]  min          Minimum scaling factor
 * @param[in]  max          Maximum scaling factor
 * @param[out] p_output     Pointer to the output scaled vector
 */
void nrf_dsp_scale_minmax_i8(const int8_t* p_input,
                             uint16_t      num,
                             const int8_t  min,
                             const int8_t  max,
                             int8_t*       p_output);

/**
 * @brief Min-Max scaling of the INT16 vector ( y = (x - min) / (max - min) )
 * @note In order not to lose accuracy when dividing, the subtraction result (x - min) is multiplied by NRF_DSP_INT16_QFACTOR
 * 
 * @param[in]  p_input      Pointer to the input vector
 * @param[in]  num          Number of samples in input vector
 * @param[in]  min          Minimum scaling factor
 * @param[in]  max          Maximum scaling factor
 * @param[out] p_output     Pointer to the output scaled vector
 */
void nrf_dsp_scale_minmax_i16(const int16_t* p_input,
                              uint16_t       num,
                              const int16_t  min,
                              const int16_t  max,
                              int16_t*       p_output);

/** @def nrf_dsp_scale_minmax_int_q
 *
 * @brief Macro for Min-Max scaling of integer variable with quantization
 * 
 * @param[in] var      Variable for scaling
 * @param[in] qfactor  Quantization factor
 * @param[in] min      Minimum scaling factor
 * @param[in] max      Maximum scaling factor
 */
#define nrf_dsp_scale_minmax_int_q(var, qfactor, min, max) \
    (((int32_t)(var - min) * qfactor) / (max - min))

/** @def nrf_dsp_scale_minmax_float
 *
 * @brief Macro for Min-Max scaling of floating-point variable
 * 
 * @param[in] var    Variable for scaling
 * @param[in] min    Minimum scaling factor
 * @param[in] max    Maximum scaling factor
 */
#define nrf_dsp_scale_minmax_float(var, min, max) ((flt32_t)(var - min) / (max - min))

/** @def nrf_dsp_scale_minmax_float_q
 *
 * @brief Macro for Min-Max scaling of floating-point variable with quantization
 * 
 * @param[in] var      Variable for scaling
 * @param[in] qfactor  Quantization factor
 * @param[in] min      Minimum scaling factor
 * @param[in] max      Maximum scaling factor
 */
#define nrf_dsp_scale_minmax_float_q(var, qfactor, min, max) (((var - min) * qfactor) / (max - min))

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SUPPORT_SCALE_MINMAX_H_ */

/**
 * @}
 */
