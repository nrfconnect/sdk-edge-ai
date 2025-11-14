/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_TYPES_H_
#define _NRF_EDGEAI_TYPES_H_

#include <nrf_edgeai/rt/nrf_edgeai_dsp_pipeline_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_input_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_model_types.h>
#include <nrf_edgeai/rt/nrf_edgeai_output_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Forward declaration of neural network context
 */
struct nrf_edgeai_s;
typedef struct nrf_edgeai_s nrf_edgeai_t;

/***********************************************************************************************************************
NRF Edge AI processing interfaces types
***********************************************************************************************************************/

/**
 * @brief Setup internal data structures for processing input features(data), 
 *          collecting data windows, etc
 *
 * @param[in, out] p_input_ctx     Pointer to the input processing context @ref nrf_edgeai_input_t 
 * 
 * @return NRF Edge AI operation status code @ref nrf_edgeai_err_t
 */
typedef nrf_edgeai_err_t (*nrf_edgeai_iface_input_setup_t)(nrf_edgeai_input_t* p_input_ctx);

/**
 * @brief Feed and collect input features to internal data structures for further processing
 * 
 * @param[in, out] p_input_ctx     Pointer to the input processing context @ref nrf_edgeai_input_t
 * @param[in] p_input_values       Input features data array
 * @param[in] num_values           Number of input features in the array
 * 
 * @return NRF Edge AI operation status code @ref nrf_edgeai_err_t
 */
typedef nrf_edgeai_err_t (*nrf_edgeai_iface_feed_inputs_t)(nrf_edgeai_input_t* p_input_ctx,
                                                           void*               p_input_values,
                                                           uint16_t            num_values);
/**
 * @brief Run model inference
 * 
 * @param[in, out] p_edgeai     Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 */
typedef void (*nrf_edgeai_iface_run_inference_t)(nrf_edgeai_t* p_edgeai);

/**
 * @brief Propagate raw model output neurons to the output values structure
 * 
 * @param[in, out] p_model    Pointer to model context @ref nrf_edgeai_model_t
 * 
 */
typedef void (*nrf_edgeai_iface_propagate_outputs_t)(nrf_edgeai_model_t* p_model);

/**
 * @brief Decode raw model outputs to human-readable format depending on the model task
 * 
 * @param[in] p_model_output      Pointer to model output structure @ref nrf_edgeai_model_t
 * @param[out] p_decoded_output   Pointer to decoded output structure @ref nrf_edgeai_decoded_output_t
 * 
 */
typedef void (*nrf_edgeai_iface_decode_outputs_t)(nrf_edgeai_model_output_t*   p_model_output,
                                                  nrf_edgeai_decoded_output_t* p_decoded_output);

/**
 * @brief Process input features for model inference, e.g. scaling, filtering, DSP, etc
 * 
 * @param[in, out] p_input        Pointer to the input processing context @ref nrf_edgeai_input_t
 * @param[in, out] p_dsp          Pointer to the DSP pipeline context @ref nrf_edgeai_dsp_pipeline_t
 * 
 * @return NRF Edge AI operation status code @ref nrf_edgeai_err_t
 */
typedef nrf_edgeai_err_t (*nrf_edgeai_iface_process_features_t)(nrf_edgeai_input_t*        p_input,
                                                                nrf_edgeai_dsp_pipeline_t* p_dsp);

/**
 * @brief Edge AI runtime interfaces structure definition
 */
typedef struct nrf_edgeai_interfaces_s
{
    nrf_edgeai_iface_input_setup_t       input_setup;
    nrf_edgeai_iface_feed_inputs_t       feed_inputs;
    nrf_edgeai_iface_process_features_t  process_features;
    nrf_edgeai_iface_run_inference_t     run_inference;
    nrf_edgeai_iface_propagate_outputs_t propagate_outputs;
    nrf_edgeai_iface_decode_outputs_t    decode_outputs;
} nrf_edgeai_interfaces_t;

/**
 * @brief EdgeAI runtime version structure
 */
typedef union nrf_edgeai_rt_version_u
{
    struct
    {
        const uint8_t  major; /**< Major version */
        const uint8_t  minor; /**< Minor version */
        const uint16_t patch; /**< Patch version */
    } field;
    const uint32_t combined; /**< Combined version as a single 32-bit value */
} nrf_edgeai_rt_version_t;

/**
 * @brief EdgeAI Solution metadata structure
 */
typedef struct nrf_edgeai_metadata_s
{
    const char*             p_solution_id; /**< Solution ID string */
    nrf_edgeai_rt_version_t version;       /**< Solution runtime version */
} nrf_edgeai_metadata_t;

/***********************************************************************************************************************
NRF Edge AI runtime context definition
***********************************************************************************************************************/
struct nrf_edgeai_s
{
    /**< Model metadata information */
    nrf_edgeai_metadata_t metadata;
    /** Input features processing context */
    nrf_edgeai_input_t input;
    /** DSP pipeline context */
    nrf_edgeai_dsp_pipeline_t* p_dsp;
    /** Model context */
    nrf_edgeai_model_t model;
    /** Decoded output results */
    nrf_edgeai_decoded_output_t decoded_output;
    /** Processing interfaces */
    nrf_edgeai_interfaces_t interfaces;
};

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_TYPES_H_ */