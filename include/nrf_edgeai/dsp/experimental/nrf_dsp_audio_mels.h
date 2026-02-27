/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_dsp_exp_audio_mels Audio Mel Spectrogram
 * @{
 * @ingroup nrf_dsp_transform
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_EXP_AUDIO_MELS_FUNCTIONS_H_
#define _NRF_DSP_EXP_AUDIO_MELS_FUNCTIONS_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize audio mel spectrogram processing pipeline
 * 
 * @param[in] sample_rate  Sampling rate of the audio input in Hz
 * @param[in] window_size_ms Window size in milliseconds
 * @param[in] window_step_size_ms Step size between windows in milliseconds
 * @param[in] mel_bins_num Number of mel frequency bins
 * @param[in] mel_fmin Minimum frequency for mel scale
 * @param[in] mel_fmax Maximum frequency for mel scale
 * 
 */
void nrf_dsp_audio_mels_init(uint16_t sample_rate,
                             uint16_t window_size_ms,
                             uint16_t window_step_size_ms,
                             uint16_t mel_bins_num,
                             uint16_t mel_fmin,
                             uint16_t mel_fmax);

/**
 * @brief Deinitialize audio mel spectrogram processing pipeline
 * 
 */
void nrf_dsp_audio_mels_deinit(void);

/**
 * @brief Process audio samples to compute mel spectrogram
 * 
 * @param[in] p_audio_samples Pointer to the input audio samples
 * @param[in] num_samples Number of audio samples
 * @param[out] p_mels Pointer to the output mel spectrogram bins
 * 
 * @return uint16_t Number of mel spectrogram bins computed
 */
uint16_t nrf_dsp_audio_mels_process(const int16_t* p_audio_samples,
                                    uint16_t       num_samples,
                                    flt32_t*       p_mels);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_EXP_AUDIO_MELS_FUNCTIONS_H_ */

/**
 * @}
 */
