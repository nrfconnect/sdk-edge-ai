/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_spectral_thd Frequencies Signal-to-Noise ratio
 * @{
 * @ingroup nrf_dsp_spectral
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SPECTRAL_FREQUENCY_SNR_H_
#define _NRF_DSP_SPECTRAL_FREQUENCY_SNR_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute signal-to-noise ratio for provided frequencies in floating-point amplitude spectrum,
 *           signal represent provided frequencies, noise is everything else
 * 
 * @param[in] p_amplitude_spectrum      Pointer to amplitude spectrum vector
 * @param[in] num                       Number of bins/samples in amplitude spectrum
 * @param[in] p_freq_indicies           Pointer to frequency indicies (bins) for which SNR will be computed
 * @param[in] freq_indicies_num         Number of frequency indicies provided
 *  
 * @return int16_t SNR of provided frequencies 
 */
flt32_t nrf_dsp_freq_snr_f32(flt32_t  p_amplitude_spectrum[],
                             uint16_t num,
                             uint16_t p_freq_indicies[],
                             uint16_t freq_indicies_num);

/**
 * @brief Compute signal-to-noise ratio for provided frequencies in fixed-point amplitude spectrum,
 *           signal represent provided frequencies, noise is everything else
 * 
 * @param[in] p_amplitude_spectrum      Pointer to amplitude spectrum vector
 * @param[in] num                       Number of bins/samples in amplitude spectrum
 * @param[in] p_freq_indicies           Pointer to frequency indicies (bins) for which SNR will be computed
 * @param[in] freq_indicies_num         Number of frequency indicies provided
 *  
 * @return int16_t SNR of provided frequencies 
 */
int16_t nrf_dsp_freq_snr_i16(int16_t  p_amplitude_spectrum[],
                             uint16_t num,
                             uint16_t p_freq_indicies[],
                             uint16_t freq_indicies_num);
#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SPECTRAL_FREQUENCY_SNR_H_ */

/**
 * @}
 */
