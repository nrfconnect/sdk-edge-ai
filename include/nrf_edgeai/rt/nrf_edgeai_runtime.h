/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_runtime nRF Edge AI Lab Runtime C-library
 * @{
 * @ingroup nrf_edgeai
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
 * nRF Edge AI runtime public API
 ***********************************************************************************************************************/

/**
 * @brief Set up the internal components of the Edge AI runtime
 *
 * @note Should be called first and once per @p p_edgeai user context
 *
 * @param[in, out] p_edgeai     Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 *
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_init(nrf_edgeai_t* p_edgeai);

/**
 * @brief  Feed raw input data to prepare it for signal processing & model inference
 *
 * @note Should be called repeatedly with new input data until the input window is filled and ready for inference.
 *      If user feed less than nrf_edgeai_input_window_size() input samples,
 *      the runtime will keep collecting input data and return NRF_EDGEAI_ERR_INPROGRESS until the window is filled.
 *
 *      If user feed more than nrf_edgeai_input_window_size() input samples,
 *      the runtime will keep the first nrf_edgeai_input_window_size() samples, remainded samples will be ignored and return NRF_EDGEAI_ERR_SUCCESS.
 *
 * @note Requires @p p_edgeai to have been successfully initialized via @ref nrf_edgeai_init() beforehand,
 *      otherwise NRF_EDGEAI_ERR_UNINITIALIZED is returned and no input data is collected.
 * 
 *      One input sample (feature vector) consists of @ref nrf_edgeai_uniq_inputs_num() individual input values (scalars),
 *      while @p num_values counts the individual values, not the samples.
 *      To feed N input samples pass num_values = N * @ref nrf_edgeai_uniq_inputs_num(),
 *      e.g. to fill a whole input window in a single call pass
 *      num_values = @ref nrf_edgeai_input_window_size() * @ref nrf_edgeai_uniq_inputs_num().
 *      e.g. if your feature vector is {x, y, z} and you want to feed 10 samples, pass num_values = 30
 *
 * @param[in, out] p_edgeai     Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * @param[in] p_input_values    Array of the input data samples,
 *                              the type of input data is dependent of neural network context nrf_edgeai_t, use @ref nrf_edgeai_input_type()
 * @param[in] num_values        Number of the individual input values(scalars) in array, should be a multiple of @ref nrf_edgeai_uniq_inputs_num()
 *
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_feed_inputs(nrf_edgeai_t* p_edgeai,
                                        void*         p_input_values,
                                        uint16_t      num_values);

/**
 * @brief Process input features for model inference, DSP processing, filtering, etc
 *
 * @note Requires @p p_edgeai to have been successfully initialized via @ref nrf_edgeai_init() and to have
 *      a full input window already collected via @ref nrf_edgeai_feed_inputs() beforehand,
 *      otherwise NRF_EDGEAI_ERR_WRONG_STATE is returned and no feature processing is performed.
 *
 * @param[in, out] p_edgeai     Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 *
 * @return Operation status code @ref nrf_edgeai_err_t
 */
nrf_edgeai_err_t nrf_edgeai_process_features(nrf_edgeai_t* p_edgeai);

/**
 * @brief Running live input features into a machine learning algorithm (or “ML/NN model”) to inference an output
 *
 * @details If the operation is succeeded (NRF_EDGEAI_ERR_SUCCESS),
 *          the inference result you can get from p_edgeai->decoded_output.regression or p_edgeai->decoded_output.classif depending on your model task
 *
 * @note Backward compatibility: if input features have not been processed yet (i.e. @ref nrf_edgeai_process_features()
 *      was not explicitly called beforehand and its state bit is not set), this function will invoke it internally
 *      before running inference. Any error returned by @ref nrf_edgeai_process_features() (e.g. NRF_EDGEAI_ERR_WRONG_STATE
 *      if the input window has not been fully collected via @ref nrf_edgeai_feed_inputs() yet) is propagated as-is.
 *      If features were already processed explicitly, that step is skipped and inference runs directly.
 *
 * @note On success, the "inputs collected" and "features processed" state bits are cleared and the
 *      "inference completed" state bit is set.
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
 * @brief Get number of model outputs (predicted targets)
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
uint16_t nrf_edgeai_model_outputs_num(const nrf_edgeai_t* p_edgeai);

/**
 * @brief Get model type @ref nrf_edgeai_model_type_t
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 */
nrf_edgeai_model_type_t nrf_edgeai_model_type(const nrf_edgeai_t* p_edgeai);

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

/**
 * @brief Get DSP feature extraction context
 * 
 * @param[in] p_edgeai  Pointer to Edge AI Lab user context @ref nrf_edgeai_t
 * 
 * @return Pointer to DSP feature extraction context @ref nrf_edgeai_dsp_feature_extraction_t
 */
const nrf_edgeai_dsp_feature_extraction_t* nrf_edgeai_dsp_features_ctx(
    const nrf_edgeai_t* p_edgeai);

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_RUNTIME_H_ */

/**
 * @}
 */
