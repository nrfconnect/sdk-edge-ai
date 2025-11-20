/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nrf_edgeai_user_model.h"
#include "nrf_edgeai_user_types.h"

#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>
#include <nrf_edgeai/nrf_edgeai_platform.h>

//////////////////////////////////////////////////////////////////////////////

#define EDGEAI_LAB_SOLUTION_ID_STR      "90508"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000001

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE f32

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE NRF_EDGEAI_INPUT_F32

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM 9

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM 9

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE 1

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT 0

/** Number of subwindows in input feature window,
* the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
* if the window size is not divisible by the number of subwindows without a remainder,
* the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM 0

#define INPUT_UNIQUE_SCALES_NUM \
    (sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0]))

//////////////////////////////////////////////////////////////////////////////
#define MODEL_NEURONS_NUM 2
#define MODEL_WEIGHTS_NUM 4
#define MODEL_OUTPUTS_NUM 1
#define MODEL_TASK        2
#define MODEL_PARAMS_TYPE f32
#define MODEL_REORDERING  0

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 1
#define MODEL_USES_AS_INPUT_DSP_FEATURES   0
#define MODEL_USES_AS_INPUT_MASK \
    ((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1))

//////////////////////////////////////////////////////////////////////////////
/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = { 0.1000000,   647.0000000, 387.0000000,
                                                             322.0000000, 559.0000000, 225.0000000,
                                                             -1.3000000,  9.1999998,   0.1847000 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = {
    11.8999996,   2040.0000000, 2214.0000000, 2683.0000000, 2775.0000000,
    2523.0000000, 44.5999985,   88.6999969,   2.1805999
};

/** Defines which unique features from the input data will be used/collected,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USAGE_MASK NULL

/** Defines which unique input features is used for LAG features processing,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USED_FOR_LAGS_MASK NULL

//////////////////////////////////////////////////////////////////////////////
#define INPUT_WINDOW_MEMORY NULL
#define P_INPUT_WINDOW_CTX  NULL

//////////////////////////////////////////////////////////////////////////////
/** The maximum number of extracted features that user used for all unique input features */
#define EXTRACTED_FEATURES_NUM 0
#define P_DSP_PIPELINE         NULL

//////////////////////////////////////////////////////////////////////////////

static const nrf_user_weight_t MODEL_WEIGHTS[] = { 0.5644180, 0.1098376, 0.9692999, -0.4583102 };

static const uint16_t MODEL_NEURONS_LINKS[] = { 2, 9, 0, 9 };

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = { 0, 3 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = { 2, 4 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = { 20.0000000, 10.0783634 };

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = { 0x1 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = { 1 };

static const nrf_user_output_t MODEL_OUTPUT_SCALE_MIN[] = { 0.2000000 };

static const nrf_user_output_t MODEL_OUTPUT_SCALE_MAX[] = { 63.7000008 };

#define NN_DECODED_OUTPUT_INIT \
.regression = {                                              \
   .meta = { .p_scale_min         = MODEL_OUTPUT_SCALE_MIN,  \
             .p_scale_max         = MODEL_OUTPUT_SCALE_MAX, }, \
}

//////////////////////////////////////////////////////////////////////////////
#define NN_INPUT_SETUP_INTERFACE       nrf_edgeai_input_setup_no_window
#define NN_INPUT_FEED_INTERFACE        nrf_edgeai_input_feed_no_window
#define NN_PROCESS_FEATURES_INTERFACE  nrf_edgeai_process_features_scale_vector_f32_f32
#define NN_RUN_INFERENCE_INTERFACE     nrf_edgeai_run_model_inference_f32
#define NN_PROPAGATE_OUTPUTS_INTERFACE nrf_edgeai_output_propagate_f32
#define NN_DECODE_OUTPUTS_INTERFACE    nrf_edgeai_output_decode_regression_f32

//////////////////////////////////////////////////////////////////////////////

static nrf_user_neuron_t model_neurons_[MODEL_NEURONS_NUM];
static nrf_user_output_t model_outputs_[MODEL_OUTPUTS_NUM];

//////////////////////////////////////////////////////////////////////////////

static nrf_edgeai_t nrf_edgeai_ = {
    ///
    .metadata.p_solution_id     = EDGEAI_LAB_SOLUTION_ID_STR,
    .metadata.version.combined  = EDGEAI_RUNTIME_VERSION_COMBINED,
    ///   
    .input.p_used_for_lags_mask = INPUT_FEATURES_USED_FOR_LAGS_MASK,
    .input.p_usage_mask         = INPUT_FEATURES_USAGE_MASK,
    .input.type                 = INPUT_FEATURE_DATA_TYPE,
    .input.unique_num           = INPUT_UNIQ_FEATURES_NUM,
    .input.unique_num_used      = INPUT_UNIQ_FEATURES_USED_NUM,
    .input.unique_scales_num    = INPUT_UNIQUE_SCALES_NUM,
    .input.window_size          = INPUT_WINDOW_SIZE,
    .input.window_shift         = INPUT_WINDOW_SHIFT,
    .input.subwindow_num        = INPUT_SUBWINDOW_NUM,
    .input.window_memory.p_void = INPUT_WINDOW_MEMORY,
    .input.p_window_ctx         = P_INPUT_WINDOW_CTX,

    .input.scale.INPUT_TYPE = {
        .p_min = INPUT_FEATURES_SCALE_MIN,
        .p_max = INPUT_FEATURES_SCALE_MAX,
    }, 
    ///
    .p_dsp = P_DSP_PIPELINE,
    ///
    .model.meta.p_neuron_internal_links_num = MODEL_NEURON_INTERNAL_LINKS_NUM,
    .model.meta.p_neuron_external_links_num = MODEL_NEURON_EXTERNAL_LINKS_NUM,
    .model.meta.p_output_neurons_indices    = MODEL_OUTPUT_NEURONS_INDICES,
    .model.meta.p_neuron_links              = MODEL_NEURONS_LINKS,
    .model.meta.p_neuron_act_type_mask      = MODEL_NEURON_ACTIVATION_TYPE_MASK,
    .model.meta.outputs_num                 = MODEL_OUTPUTS_NUM,
    .model.meta.neurons_num                 = MODEL_NEURONS_NUM,
    .model.meta.weights_num                 = MODEL_WEIGHTS_NUM,
    .model.meta.task                        = MODEL_TASK,
    .model.meta.uses_as_input.all           = MODEL_USES_AS_INPUT_MASK,

    .model.params.MODEL_PARAMS_TYPE = {
        .p_weights      = MODEL_WEIGHTS,
        .p_act_weights = MODEL_NEURON_ACTIVATION_WEIGHTS,
        .p_neurons      = model_neurons_,
    },

    .model.output.memory.p_void = model_outputs_,
    .model.output.num = MODEL_OUTPUTS_NUM,
    ///
    .interfaces.input_setup = NN_INPUT_SETUP_INTERFACE,
    .interfaces.feed_inputs = NN_INPUT_FEED_INTERFACE,
    .interfaces.process_features = NN_PROCESS_FEATURES_INTERFACE,
    .interfaces.run_inference = NN_RUN_INFERENCE_INTERFACE,
    .interfaces.propagate_outputs = NN_PROPAGATE_OUTPUTS_INTERFACE,
    .interfaces.decode_outputs = NN_DECODE_OUTPUTS_INTERFACE,
    ///
    .decoded_output = { NN_DECODED_OUTPUT_INIT },
};

//////////////////////////////////////////////////////////////////////////////

nrf_edgeai_t* nrf_edgeai_user_model(void)
{
    return &nrf_edgeai_;
}

//////////////////////////////////////////////////////////////////////////////

uint32_t nrf_edgeai_user_model_size(void)
{
    uint32_t model_meta_size =
        (sizeof(MODEL_WEIGHTS) + sizeof(MODEL_NEURONS_LINKS) +
         sizeof(MODEL_NEURON_EXTERNAL_LINKS_NUM) + sizeof(MODEL_NEURON_INTERNAL_LINKS_NUM) +
         sizeof(MODEL_NEURON_ACTIVATION_WEIGHTS) + sizeof(MODEL_NEURON_ACTIVATION_TYPE_MASK) +
         sizeof(MODEL_OUTPUT_NEURONS_INDICES));

#if MODEL_TASK == __NRF_EDGEAI_TASK_ANOMALY_DETECTION
    model_meta_size += sizeof(MODEL_AVERAGE_EMBEDDING) + sizeof(MODEL_OUTPUT_SCALE_MIN) +
                       sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

#if MODEL_TASK == __NRF_EDGEAI_TASK_REGRESSION
    model_meta_size += sizeof(MODEL_OUTPUT_SCALE_MIN) + sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

    return model_meta_size;
}
