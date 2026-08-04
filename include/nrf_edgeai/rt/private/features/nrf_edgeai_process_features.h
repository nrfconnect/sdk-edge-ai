/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#ifndef _NRF_EDGEAI_PRIVATE_INTERFACES_PROCESS_FEATURES_H_
#define _NRF_EDGEAI_PRIVATE_INTERFACES_PROCESS_FEATURES_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @name DSP Feature Extraction
 *  @brief Extract DSP features from input data.
 *
 *  These functions perform digital signal processing (DSP) feature extraction (such as statistical or spectral features)
 *  from the input data.
 *  The extraction is performed according to the configuration in the neural network context.
 * 
 * @note For models that do not require feature extraction, an empty implementation is provided.
 *
 *  @param[in, out] p_input        Pointer to the input processing context @ref nrf_edgeai_input_t
 *  @param[in, out] p_dsp          Pointer to the DSP pipeline context @ref nrf_edgeai_dsp_pipeline_t
 *  @return Status code indicating success or error.
 * @{
 */

nrf_edgeai_err_t nrf_edgeai_process_features_dsp_i8(nrf_edgeai_input_t*        p_input,
                                                    nrf_edgeai_dsp_pipeline_t* p_dsp);

nrf_edgeai_err_t nrf_edgeai_process_features_dsp_i16(nrf_edgeai_input_t*        p_input,
                                                     nrf_edgeai_dsp_pipeline_t* p_dsp);

nrf_edgeai_err_t nrf_edgeai_process_features_dsp_f32(nrf_edgeai_input_t*        p_input,
                                                     nrf_edgeai_dsp_pipeline_t* p_dsp);

nrf_edgeai_err_t nrf_edgeai_process_features_empty(nrf_edgeai_input_t*        p_input,
                                                   nrf_edgeai_dsp_pipeline_t* p_dsp);
/** @} */
#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_PRIVATE_INTERFACES_PROCESS_FEATURES_H_ */