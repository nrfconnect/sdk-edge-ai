/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 *
 * @defgroup nrf_edgeai NRF Edge AI Lab Runtime C-library
 * @{
 *
 * @details The interface library for Nordic EdgeAI Lab solutions processing
 *
 */

#ifndef _NRF_EDGEAI_RUNTIME_H_
#define _NRF_EDGEAI_RUNTIME_H_

#include <nrf_edgeai/rt/nrf_edgeai_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/***********************************************************************************************************************
 * NRF Edge AI runtime public API
 ***********************************************************************************************************************/

/**
 * @brief Set up the internal components of the Edge AI runtime 
 *        should be called first and once
 * 
 * @param[in, out] p_edgeai     Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 *  
 * @return Operation status code @ref nrf_edgeai_err_t 
 */
nrf_edgeai_err_t nrf_edgeai_init(nrf_edgeai_t* p_edgeai);

/**
 * @brief  Feed raw input data to prepare it for signal processing & model inference
 * 
 * @param[in, out] p_edgeai     Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * @param[in] p_input_values    Array of the input data samples, 
 *                              the type of input data is dependent of neural network context nrf_edgeai_t, use @ref nrf_edgeai_input_type()
 * @param[in] num_values        Number of the input samples in array, should be a multiple of @ref nrf_edgeai_uniq_inputs_num()
 * 
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_feed_inputs(nrf_edgeai_t* p_edgeai,
                                        void*         p_input_values,
                                        uint16_t      num_values);

/**
 * @brief Running live input features into a machine learning algorithm (or “ML/NN model”) to inference an output
 * 
 * @details If the operation is succeeded (NRF_EDGEAI_ERR_SUCCESS), 
 *          the inference result you can get from p_edgeai->decoded_output.regression or p_edgeai->decoded_output.classif depending on your model task
 * 
 * @param[in, out] p_edgeai     Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 * @return Operation status code @ref nrf_edgeai_err_t 
 */
nrf_edgeai_err_t nrf_edgeai_run_inference(nrf_edgeai_t* p_edgeai);

/***********************************************************************************************************************
Utility variables and functions
***********************************************************************************************************************/

/**
 * @brief Get neural network input data type @ref nrf_edgeai_input_type_t 
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 */
nrf_edgeai_input_type_t nrf_edgeai_input_type(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get number of unique input features on which the model was trained,
 *         e.g for features {x, y, z} -> number of unique input features = 3
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 *  
 */
uint16_t nrf_edgeai_uniq_inputs_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get input features window size in feature samples(vectors),
 *         e.g for input window {x0, y0, z0, ..., xn, yn, zn} -> window size = n
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 */
uint16_t nrf_edgeai_input_window_size(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get number of subwindows in the input window
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 */
uint8_t nrf_edgeai_input_subwindows_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get number of model neurons
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
uint16_t nrf_edgeai_model_neurons_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get number of model weights
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
uint16_t nrf_edgeai_model_weights_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get number of model outputs (predicted targets)
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
uint16_t nrf_edgeai_model_outputs_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get model task @ref nrf_edgeai_model_task_t
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
nrf_edgeai_model_task_t nrf_edgeai_model_task(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get solution ID in string format
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
const char* nrf_edgeai_solution_id_str(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get solution runtime version
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
nrf_edgeai_rt_version_t nrf_edgeai_solution_runtime_version(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get Edge AI runtime library version
 * 
 */
nrf_edgeai_rt_version_t nrf_edgeai_runtime_version(void);

/**
 * @brief Check if the Edge AI runtime library version is compatible with the solution runtime version
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 * @return true if compatible, false otherwise
 */
bool nrf_edgeai_is_runtime_compatible(const nrf_edgeai_t* p_edgeai);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_RUNTIME_H_ */

/**
 * @}
 */
