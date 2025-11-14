/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_spectral_centroid Spectral Centroid
 * @{
 * @ingroup nrf_dsp_spectral
 *
 * @brief The spectral centroid represents the "center of mass" of the frequency content in a vibration signal,
 *        indicating the average frequency where most of the signal's energy is concentrated. 
 *        It provides insight into the distribution of energy across different frequency bands.
 *
 */
#ifndef _NRF_DSP_SPECTRAL_CENTROID_H_
#define _NRF_DSP_SPECTRAL_CENTROID_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Spectral Centroid in floating-point spectrum
 * 
 * @param[in]       p_spectrum            Pointer to amplitude spectrum vector
 * @param[in]       num                   Number of bins/samples in amplitude spectrum
 * @param[in, out]  p_ctx                 Pointer to optional spectral context,
 *                                        the calculated value of magnitude_sum and cetroid will be written to context for further use
 *  
 * @return flt32_t Spectral Centroid
 */
flt32_t nrf_dsp_spectral_centroid_f32(flt32_t                     p_spectrum[],
                                      uint16_t                    num,
                                      nrf_dsp_spectral_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Spectral Centroid in INT16 fixed-point spectrum
 * 
 * @param[in]       p_spectrum            Pointer to amplitude spectrum vector
 * @param[in]       num                   Number of bins/samples in amplitude spectrum
 * @param[in, out]  p_ctx                 Pointer to optional spectral context,
 *                                        the calculated value of magnitude_sum and cetroid will be written to context for further use
 *  
 * @return int16_t Spectral Centroid
 */
int16_t nrf_dsp_spectral_centroid_i16(int16_t                     p_spectrum[],
                                      uint16_t                    num,
                                      nrf_dsp_spectral_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SPECTRAL_CENTROID_H_ */

/**
 * @}
 */
