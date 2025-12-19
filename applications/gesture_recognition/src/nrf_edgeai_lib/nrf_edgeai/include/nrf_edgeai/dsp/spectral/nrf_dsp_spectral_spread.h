/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_spectral_centroid Spectral Spread
 * @{
 * @ingroup nrf_dsp_spectral
 *
 * @brief Spectral spreads represent the standard deviation of amplitude distribution around a sound’s spectral centroid, 
 *          calculated using the square root of its second-order moment. 
 *          A signal with a high spectral spread will look “larger” and “flatter” (literally more “spread”) 
 *          than a signal with a lower spectral spread value.
 *
 */
#ifndef _NRF_DSP_SPECTRAL_SPREAD_H_
#define _NRF_DSP_SPECTRAL_SPREAD_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Calculate Spectral Spread in floating-point spectrum
 * 
 * @param[in]       p_spectrum            Pointer to amplitude spectrum vector
 * @param[in]       num                   Number of bins/samples in amplitude spectrum
 * @param[in, out]  p_ctx                 Pointer to optional spectral context,
 *                                        the calculated value of magnitude_sum and cetroid will be used
 *  
 * @return flt32_t Spectral Spread
 */
flt32_t nrf_dsp_spectral_spread_f32(flt32_t                     p_spectrum[],
                                    uint16_t                    num,
                                    nrf_dsp_spectral_ctx_f32_t* p_ctx);

/**
 * @brief Calculate Spectral Spread in INT16 fixed-point spectrum
 * 
 * @param[in]       p_spectrum            Pointer to amplitude spectrum vector
 * @param[in]       num                   Number of bins/samples in amplitude spectrum
 * @param[in, out]  p_ctx                 Pointer to optional spectral context,
 *                                        the calculated value of magnitude_sum and cetroid will be used
 *  
 * @return int16_t Spectral Spread
 */
int16_t nrf_dsp_spectral_spread_i16(int16_t                     p_spectrum[],
                                    uint16_t                    num,
                                    nrf_dsp_spectral_ctx_i16_t* p_ctx);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SPECTRAL_SPREAD_H_ */

/**
 * @}
 */
