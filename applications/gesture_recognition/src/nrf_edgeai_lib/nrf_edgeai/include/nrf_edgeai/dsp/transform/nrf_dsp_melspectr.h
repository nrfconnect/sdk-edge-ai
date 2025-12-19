/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_dsp_transform_melspec Mel-Spectrogram
 * @{
 * @ingroup nrf_dsp_transform
 *
 * @brief
 *
 */
#ifndef _NRF_DSP_TRANSFORM_MELSPECTROGRAM_H_
#define _NRF_DSP_TRANSFORM_MELSPECTROGRAM_H_

#include <nrf_edgeai/dsp/nrf_dsp_types.h>
#include "nrf_dsp_rfht.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Structure descibing mel-spectrogram preprocessing block */
typedef struct
{
    /** Buffer to store mel-spectrogram 
     * (elements count is @mel_spectrum_n_mels * @mel_spectrum_count)
     */
    flt32_t* p_melspectrum;

    /** FHT instance */
    nrf_dsp_rfht_instance_t fht;

    /** Sample rate of audio signal */
    uint16_t sample_rate;

    /** Current spectrogram fill */
    uint16_t current_fill;

    /** Count of frequency bins */
    uint16_t freq_bands;

    /** Spectrogram Time Bands dim */
    uint16_t time_bands;
} nrf_dsp_melspectr_ctx_f32_t;

/**
 * @brief Make mel-spectrogram from timeseries audio data
 * 
 * @param[in] p_ctx            Pointer to mel-spectrogram instance
 * @param[in] p_audio_input    Audio input to process
 * 
 * @return int8_t      Return 0 if spectrogram is ready, otherwise 1
 */
int8_t nrf_dsp_melspectr_make_f32(nrf_dsp_melspectr_ctx_f32_t* p_ctx, flt32_t* p_audio_input);

/**
 * @brief Shift mel-spectrogram to the right on N columns
 * 
 * @param[in] p_ctx        Pointer to mel-spectrogram instance
 * @param[in] shift        Number of spectrograms to shift
 */
void nrf_dsp_melspectr_shift_f32(nrf_dsp_melspectr_ctx_f32_t* p_ctx, uint16_t shift);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_DSP_TRANSFORM_MELSPECTROGRAM_H_ */

/**
 * @}
 */
