/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_spectral_thd Frequency Total Harmonic Distortion
 * @{
 * @ingroup nrf_dsp_spectral
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_SPECTRAL_FREQUENCY_THD_H_
#define _NRF_DSP_SPECTRAL_FREQUENCY_THD_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute Total Harmonic Distortion for provided frequency in floating point spectrum
 * 
 * @param[in] p_amplitude_spectrum      Pointer to amplitude spectrum vector
 * @param[in] num                       Number of bins/samples in amplitude spectrum
 * @param[in] base_freq_index           Frequency index(bin) in amplitude spectrum for which THD will be computed
 * @param[in] n_harmonics               Number of harmonics to compute THD, starting from second one, 
 *                                      e.g. n_harmonics = 3 then for THD will be used H_2x, H_3x, H_4x
 *  
 * @return flt32_t Total Harmonic Distortion
 */
flt32_t nrf_dsp_freq_thd_f32(flt32_t  p_amplitude_spectrum[],
                             uint16_t num,
                             uint16_t base_freq_index,
                             uint16_t n_harmonics);

/**
 * @brief Compute Total Harmonic Distortion for provided frequency in fixed-point spectrum
 * 
 * @param[in] p_amplitude_spectrum      Pointer to amplitude spectrum vector
 * @param[in] num                       Number of bins/samples in amplitude spectrum
 * @param[in] base_freq_index           Frequency index(bin) in amplitude spectrum for which THD will be computed
 * @param[in] n_harmonics               Number of harmonics to compute THD, starting from second one, 
 *                                      e.g. n_harmonics = 3 then for THD will be used H_2x, H_3x, H_4x
 *  
 * @return int16_t Relative Total Harmonic Distortion
 */
int16_t nrf_dsp_freq_thd_i16(int16_t  p_amplitude_spectrum[],
                             uint16_t num,
                             uint16_t base_freq_index,
                             uint16_t n_harmonics);
#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_SPECTRAL_FREQUENCY_THD_H_ */

/**
 * @}
 */
