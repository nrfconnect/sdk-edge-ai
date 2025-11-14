/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_statistic_pk2pk Peak to Peak 
 * @{
 * @ingroup nrf_dsp_statistic
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_STAT_PEAK2PEAK_FUNCTIONS_H_
#define _NRF_DSP_STAT_PEAK2PEAK_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Finds Peak-to-Peak amplitude for Low and High frequency components of signal, 
 *          represented as a floating-point vector.
 * 
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * @param[out]  p_pk2pk_lf      Pointer to store Peak-to-Peak amplitude for Low frequency component
 * @param[out]  p_pk2pk_hf      Pointer to store Peak-to-Peak amplitude for High frequency component
 */
void nrf_dsp_pk2pk_lf_hf_f32(const flt32_t* p_input,
                             uint16_t       num,
                             size32_t       window_size,
                             flt32_t*       p_pk2pk_lf,
                             flt32_t*       p_pk2pk_hf);

/**
 * @brief Finds Peak-to-Peak amplitude for Low and High frequency components of signal, 
 *          represented as a floating-point vector, using values ​​in increments of 'stride'.
 * 
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * @param[out]  p_pk2pk_lf      Pointer to store Peak-to-Peak amplitude for Low frequency component
 * @param[out]  p_pk2pk_hf      Pointer to store Peak-to-Peak amplitude for High frequency component
 */
void nrf_dsp_pk2pk_lf_hf_f32_s(const flt32_t* p_input,
                               uint16_t       num,
                               size32_t       stride,
                               size32_t       window_size,
                               flt32_t*       p_pk2pk_lf,
                               flt32_t*       p_pk2pk_hf);

/**
 * @brief Finds Peak-to-Peak amplitude for Low and High frequency components of signal, 
 *          represented as a INT8 fixed-point vector.
 * 
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * @param[out]  p_pk2pk_lf      Pointer to store Peak-to-Peak amplitude for Low frequency component
 * @param[out]  p_pk2pk_hf      Pointer to store Peak-to-Peak amplitude for High frequency component
 */
void nrf_dsp_pk2pk_lf_hf_i8(const int8_t* p_input,
                            uint16_t      num,
                            size32_t      window_size,
                            int16_t*      p_pk2pk_lf,
                            int16_t*      p_pk2pk_hf);

/**
 * @brief Finds Peak-to-Peak amplitude for Low and High frequency components of signal, 
 *          represented as a INT8 fixed-point vector, using values ​​in increments of 'stride'.
 * 
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * @param[out]  p_pk2pk_lf      Pointer to store Peak-to-Peak amplitude for Low frequency component
 * @param[out]  p_pk2pk_hf      Pointer to store Peak-to-Peak amplitude for High frequency component
 */
void nrf_dsp_pk2pk_lf_hf_i8_s(const int8_t* p_input,
                              uint16_t      num,
                              size32_t      stride,
                              size32_t      window_size,
                              int16_t*      p_pk2pk_lf,
                              int16_t*      p_pk2pk_hf);

/**
 * @brief Finds Peak-to-Peak amplitude for Low and High frequency components of signal, 
 *          represented as a INT16 fixed-point vector.
 * 
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * @param[out]  p_pk2pk_lf      Pointer to store Peak-to-Peak amplitude for Low frequency component
 * @param[out]  p_pk2pk_hf      Pointer to store Peak-to-Peak amplitude for High frequency component
 */
void nrf_dsp_pk2pk_lf_hf_i16(const int16_t* p_input,
                             uint16_t       num,
                             size32_t       window_size,
                             int32_t*       p_pk2pk_lf,
                             int32_t*       p_pk2pk_hf);

/**
 * @brief Finds Peak-to-Peak amplitude for Low and High frequency components of signal, 
 *          represented as a INT16 fixed-point vector, using values ​​in increments of 'stride'.
 * 
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * @param[out]  p_pk2pk_lf      Pointer to store Peak-to-Peak amplitude for Low frequency component
 * @param[out]  p_pk2pk_hf      Pointer to store Peak-to-Peak amplitude for High frequency component
 */
void nrf_dsp_pk2pk_lf_hf_i16_s(const int16_t* p_input,
                               uint16_t       num,
                               size32_t       stride,
                               size32_t       window_size,
                               int32_t*       p_pk2pk_lf,
                               int32_t*       p_pk2pk_hf);

/**
 * @brief Finds Peak-to-Peak amplitude for High frequency component of signal, 
 *          represented as a floating-point vector.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  flt32_t  Peak-to-Peak amplitude for High frequency component 
 */
flt32_t nrf_dsp_pk2pk_hf_f32(const flt32_t* p_input, uint16_t num, size32_t window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for High frequency component of signal, 
 *          represented as a floating-point vector, using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  flt32_t  Peak-to-Peak amplitude for High frequency component 
 */
flt32_t nrf_dsp_pk2pk_hf_f32_s(const flt32_t* p_input,
                               uint16_t       num,
                               size32_t       stride,
                               size32_t       window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for High frequency component of signal, 
 *          represented as a INT8 fixed-point vector.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int16_t  Peak-to-Peak amplitude for High frequency component 
 */
int16_t nrf_dsp_pk2pk_hf_i8(const int8_t* p_input, uint16_t num, size32_t window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for High frequency component of signal, 
 *          represented as a INT8 fixed-point vector, using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int16_t  Peak-to-Peak amplitude for High frequency component 
 */
int16_t nrf_dsp_pk2pk_hf_i8_s(const int8_t* p_input,
                              uint16_t      num,
                              size32_t      stride,
                              size32_t      window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for High frequency component of signal, 
 *          represented as a INT16 fixed-point vector.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int32_t  Peak-to-Peak amplitude for High frequency component 
 */
int32_t nrf_dsp_pk2pk_hf_i16(const int16_t* p_input, uint16_t num, size32_t window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for High frequency component of signal, 
 *          represented as a INT16 fixed-point vector, using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int32_t  Peak-to-Peak amplitude for High frequency component 
 */
int32_t nrf_dsp_pk2pk_hf_i16_s(const int16_t* p_input,
                               uint16_t       num,
                               size32_t       stride,
                               size32_t       window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for Low frequency component of signal, 
 *          represented as a floating-point vector.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  flt32_t  Peak-to-Peak amplitude for Low frequency component 
 */
flt32_t nrf_dsp_pk2pk_lf_f32(const flt32_t* p_input, uint16_t num, size32_t window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for Low frequency component of signal, 
 *          represented as a floating-point vector, using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  flt32_t  Peak-to-Peak amplitude for Low frequency component 
 */
flt32_t nrf_dsp_pk2pk_lf_f32_s(const flt32_t* p_input,
                               uint16_t       num,
                               size32_t       stride,
                               size32_t       window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for Low frequency component of signal, 
 *          represented as a INT8 fixed-point vector.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int16_t  Peak-to-Peak amplitude for Low frequency component 
 */
int16_t nrf_dsp_pk2pk_lf_i8(const int8_t* p_input, uint16_t num, size32_t window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for Low frequency component of signal, 
 *          represented as a INT8 fixed-point vector, using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int16_t  Peak-to-Peak amplitude for Low frequency component 
 */
int16_t nrf_dsp_pk2pk_lf_i8_s(const int8_t* p_input,
                              uint16_t      num,
                              size32_t      stride,
                              size32_t      window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for Low frequency component of signal, 
 *          represented as a INT16 fixed-point vector.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples in input vector
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int32_t  Peak-to-Peak amplitude for Low frequency component 
 */
int32_t nrf_dsp_pk2pk_lf_i16(const int16_t* p_input, uint16_t num, size32_t window_size);

/**
 * @brief Finds Peak-to-Peak amplitude for Low frequency component of signal, 
 *          represented as a INT16 fixed-point vector, using values ​​in increments of 'stride'.
 *
 * @param[in]   p_input         Pointer to the input vector
 * @param[in]   num             Number of samples with 'stride' in input vector
 * @param[in]   stride          Vector element offset stride
 * @param[in]   window_size     Determines window size for calculation moving average filter sum,
 *                              window_size affect attenuation for frequencies
 * 
 * @return  int32_t  Peak-to-Peak amplitude for Low frequency component 
 */
int32_t nrf_dsp_pk2pk_lf_i16_s(const int16_t* p_input,
                               uint16_t       num,
                               size32_t       stride,
                               size32_t       window_size);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_STAT_PEAK2PEAK_FUNCTIONS_H_ */

/**
 * @}
 */
