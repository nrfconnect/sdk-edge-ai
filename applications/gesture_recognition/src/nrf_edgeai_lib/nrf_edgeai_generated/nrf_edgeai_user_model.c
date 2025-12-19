/* 2025-12-03T11:23:19.063018 */
/*
* Copyright (c) 2021 Nordic Semiconductor ASA
* SPDX-License-Identifier: Apache-2.0
*/
#include "nrf_edgeai_user_model.h"
#include "nrf_edgeai_user_types.h"
#include <nrf_edgeai/nrf_edgeai_platform.h>
#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>

//////////////////////////////////////////////////////////////////////////////
/* Nordic EdgeAI Lab Solution ID and Runtime Version */
#define EDGEAI_LAB_SOLUTION_ID_STR      "84622"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000001

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE                         i16

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE            NRF_EDGEAI_INPUT_I16

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM            6

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM       6

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE                  99

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT                 33

/** Number of subwindows in input feature window,
* the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
* if the window size is not divisible by the number of subwindows without a remainder,
* the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM                 0

#define INPUT_UNIQUE_SCALES_NUM (sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0])) 

//////////////////////////////////////////////////////////////////////////////
#define MODEL_SOLUTION_ID_STR      "84622"
#define MODEL_NEURONS_NUM          25
#define MODEL_WEIGHTS_NUM          217
#define MODEL_OUTPUTS_NUM          8
#define MODEL_TASK                 0
#define MODEL_PARAMS_TYPE          q16
#define MODEL_REORDERING           0

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 0
#define MODEL_USES_AS_INPUT_DSP_FEATURES 1
#define MODEL_USES_AS_INPUT_MASK ((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1)) 

//////////////////////////////////////////////////////////////////////////////
/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = {
 -32768 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = {
 32767 };

/** Defines which unique features from the input data will be used/collected,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USAGE_MASK NULL

/** Defines which unique input features is used for LAG features processing,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USED_FOR_LAGS_MASK NULL

//////////////////////////////////////////////////////////////////////////////
/** Input feature buffer element size, 
 * if quantization of model is bigger than input features size in bits, 
 * the size of input buffer should aligned to nrf_user_neuron_t */ 
#define INPUT_TYPE_SIZE \
    ((sizeof(nrf_user_input_t) > sizeof(nrf_user_neuron_t)) ? sizeof(nrf_user_input_t) : sizeof(nrf_user_neuron_t)) 

/** Input features window size in bytes to allocate statically */ 
#define INPUT_WINDOW_BUFFER_SIZE_BYTES \
    (INPUT_WINDOW_SIZE * INPUT_UNIQ_FEATURES_NUM * INPUT_TYPE_SIZE) 

static uint8_t input_window_[INPUT_WINDOW_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED; 

#define INPUT_WINDOW_MEMORY    &input_window_[0] 

static nrf_edgeai_window_ctx_t input_window_ctx_; 
#define P_INPUT_WINDOW_CTX     &input_window_ctx_ 

//////////////////////////////////////////////////////////////////////////////
/** The maximum number of extracted features that user used for all unique input features */
#define EXTRACTED_FEATURES_NUM 52

#define EXTRACTED_FEATURES_META_TYPE i32 

/** DSP feature buffer element size,
 * if quantization of model is bigger than DSP features size in bits,
 * the size of extracted DSP features buffer should aligned to nrf_user_neuron_t */
#define EXTRACTED_FEATURE_SIZE_BYTES                                                  \
    ((sizeof(nrf_user_feature_t) > sizeof(nrf_user_neuron_t)) ? sizeof(nrf_user_feature_t) : \
                                                            sizeof(nrf_user_neuron_t))

/** Size of extracted features buffer in bytes */
#define EXTRACTED_FEATURES_BUFFER_SIZE_BYTES (EXTRACTED_FEATURES_NUM * EXTRACTED_FEATURE_SIZE_BYTES) 

/** Defines feature extraction masks used as nrf_edgeai_features_mask_t,
 *  64 bit for one unique input feature, @ref nrf_edgeai_features_mask_t to see bitmask
 */

static const uint64_t FEATURES_EXTRACTION_MASK[] = { 0x0044c19b00000000,
     0x0044c39b00000000, 0x0044419300000000, 0x0040c19b00000000, 0x0000c69300000000,
     0x0040811300000000 };
/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
#define FEATURES_EXTRACTION_ARGUMENTS NULL

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = { -32768, -8042,
     -11225, 12, 17, 60, 48, 13, 0, 167, -32768, -16364, -19331, 11, 14, 30,
     10, 25, 15, 0, 192, -32768, -8146, 12, 20, 30, 24, 0, 202, -32768, -2837,
     -6946, 2, 4, 7, 5, 3, 50, -32768, -102, 3, 4, 10, 0, 30, 4, -32768, -2065,
     3, 7, 3, 50 };

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = { 14360, 32767,
     17104, 21566, 24224, 24712, 22589, 6150, 1000, 65377, 15840, 32767, 16027,
     20098, 21927, 21963, 602, 20101, 8953, 1000, 64936, 15672, 32767, 14635,
     16460, 17738, 16641, 1000, 65366, 2185, 32767, 6406, 19454, 21626, 21737,
     19502, 3197, 44192, 593, 32767, 20949, 23215, 612, 387, 21019, 4365, 1052,
     32767, 23261, 24547, 4055, 56197 };

/** Memory allocation to store extracted features during DSP pipeline */
static uint8_t extracted_features_buffer_[EXTRACTED_FEATURES_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED;


/** Timedomain features processing context  */
#define P_TIMEDOMAIN_FEATURES_CTX  NULL
/** Timedomain features in feature extraction pipeline  */
static const nrf_edgeai_features_pipeline_func_i16_t timedomain_features_[] = {
    nrf_edgeai_feature_utility_tss_sum_i16,
    nrf_edgeai_feature_min_max_range_i16,
    nrf_edgeai_feature_mean_i16,
    nrf_edgeai_feature_mad_i16,
    nrf_edgeai_feature_std_i16,
    nrf_edgeai_feature_rms_i16,
    nrf_edgeai_feature_mcr_i16,
    nrf_edgeai_feature_zcr_i16,
    nrf_edgeai_feature_absmean_i16,
    nrf_edgeai_feature_amdf_i16,
    nrf_edgeai_feature_psoz_i16,
    nrf_edgeai_feature_rmds_i16,
 };

static const nrf_edgeai_features_pipeline_ctx_t timedomain_pipeline_ = {
    .functions_num     = sizeof(timedomain_features_) / sizeof(timedomain_features_[0]),
    .functions.p_void  = timedomain_features_,
    .p_ctx             = P_TIMEDOMAIN_FEATURES_CTX,
};
#define P_TIMEDOMAIN_PIPELINE &timedomain_pipeline_ 

#define P_FREQDOMAIN_PIPELINE NULL

static nrf_edgeai_dsp_pipeline_t dsp_pipeline_ = { 
   .features = {  
       .p_masks = (nrf_edgeai_features_mask_t*)FEATURES_EXTRACTION_MASK, 
       .extracted_memory.p_void = extracted_features_buffer_, 
       .overall_num = EXTRACTED_FEATURES_NUM, 
       .masks_num = sizeof(FEATURES_EXTRACTION_MASK) / sizeof(FEATURES_EXTRACTION_MASK[0]), 

       .p_timedomain_pipeline = P_TIMEDOMAIN_PIPELINE, 
       .p_freqdomain_pipeline = P_FREQDOMAIN_PIPELINE, 

       .meta.EXTRACTED_FEATURES_META_TYPE = { 
           .p_min = EXTRACTED_FEATURES_SCALE_MIN, 
           .p_max = EXTRACTED_FEATURES_SCALE_MAX, 
       .p_arguments = FEATURES_EXTRACTION_ARGUMENTS, 
       },
   }, 
}; 

#define P_DSP_PIPELINE         &dsp_pipeline_ 


//////////////////////////////////////////////////////////////////////////////

static const nrf_user_weight_t MODEL_WEIGHTS[] = { -7401, 13914, 32763, -6046,
     -32650, 30028, 32524, -3292, 32766, 32763, 7111, 28900, -8125, 32764, -24595,
     2163, 7738, -10988, 11126, -2335, 1608, 7764, -12598, 31040, -19110, -13618,
     16645, -3109, -1199, -26426, -3921, -13086, 22524, 26586, 6409, -21586,
     -17110, 5597, -32742, 19131, 22626, -21029, 2059, 20728, 11375, 8262, 16286,
     12182, 5790, -9328, 32764, 22129, -31923, 32765, 28718, -9757, -18819, 20804,
     15370, -31521, -19166, 30113, 13603, -11641, 1374, -22727, 32767, -7307,
     -24660, 2358, 26347, 6422, -23461, 25528, -15937, -30129, 14018, -31022,
     1916, 32101, 25315, -5784, -9927, -23502, 10306, 32767, -22337, -10526,
     23779, 6467, -13962, 2656, 29231, 32766, 32384, 29734, -2680, 25143, -346,
     15715, -12290, -2918, 32765, 32528, -23776, 18732, 27568, 28236, -3564,
     32765, 11179, -23799, 11250, 24618, 5585, -6563, -32659, 11617, -16574,
     -32017, 322, -5892, 3053, 17892, 8367, 4631, -4756, 14113, 3428, 27747,
     -188, 15126, 11119, 9037, -16383, -32766, -22422, -32768, 32766, 1631, 31724,
     -9744, 32765, 32767, -16114, 32766, -12645, -32768, -32768, 21292, -32766,
     2695, 32766, 12631, 9809, -17148, -2178, 32765, -32768, -8679, -16009, 32553,
     32766, 2993, 8029, 25829, 16383, 28732, 32767, -32592, 32766, 2879, -29122,
     -32768, 21382, -28258, 19790, 23479, 13460, 32465, -17727, 11680, -28806,
     -32768, -32768, 8477, 6479, -31138, -26818, 8443, 9583, -8451, -1060, 325,
     -9489, 9472, -10801, 17917, -3818, -32768, 32766, -12, 1331, 4752, -32766,
     -3661, 20376, -27057, -32766, 2937, -11833, 22932, -17513, -31973, 32765,
     -32768, 9854 };

static const uint16_t MODEL_NEURONS_LINKS[] = { 0, 1, 18, 24, 29, 30, 33,
     35, 48, 49, 50, 52, 2, 5, 6, 7, 11, 12, 15, 17, 20, 25, 32, 46, 50, 52,
     1, 0, 17, 19, 21, 33, 38, 39, 45, 47, 52, 0, 1, 2, 2, 4, 8, 12, 13, 15,
     18, 21, 23, 27, 38, 39, 41, 45, 51, 52, 0, 3, 4, 6, 10, 11, 14, 17, 18,
     19, 20, 22, 26, 28, 30, 37, 41, 42, 43, 44, 47, 52, 0, 5, 6, 7, 9, 11, 12,
     19, 20, 22, 25, 26, 27, 36, 39, 48, 49, 52, 1, 1, 2, 4, 5, 8, 9, 12, 13,
     16, 30, 31, 33, 42, 45, 52, 6, 6, 8, 13, 31, 52, 1, 4, 6, 0, 2, 7, 11, 12,
     17, 35, 52, 1, 0, 5, 10, 15, 40, 41, 52, 0, 9, 52, 2, 52, 2, 11, 52, 3,
     52, 3, 13, 52, 4, 5, 9, 1, 29, 43, 52, 4, 15, 52, 1, 5, 9, 6, 10, 15, 19,
     25, 44, 46, 48, 52, 5, 17, 52, 7, 5, 16, 29, 31, 33, 51, 52, 6, 19, 52,
     2, 7, 8, 19, 1, 7, 9, 15, 21, 31, 34, 40, 52, 7, 21, 52, 1, 6, 8, 17, 10,
     12, 17, 22, 24, 46, 52, 1, 8, 23, 52 };

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = { 0, 12,
     27, 40, 56, 78, 97, 113, 121, 130, 139, 141, 144, 146, 149, 153, 159, 163,
     174, 176, 185, 190, 201, 206, 216 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = { 12, 26,
     37, 56, 78, 96, 112, 118, 129, 137, 140, 142, 145, 147, 150, 157, 160, 172,
     175, 183, 186, 199, 202, 213, 217 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = { 0, 0, 0, 0, 0, 0, 0,
     0, 11264, 64512, 40959, 60416, 40959, 63488, 40960, 60416, 40955, 16384,
     40960, 0, 40960, 60416, 40954, 64512, 40959 };

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = { 0xff, 0xab, 0xaa, 0x00 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = { 10, 24, 12, 14,
     16, 18, 20, 22 };

#define NN_DECODED_OUTPUT_INIT                 \
.classif = {                                   \
   .predicted_class = 0,                       \
   .num_classes = MODEL_OUTPUTS_NUM,           \
}

//////////////////////////////////////////////////////////////////////////////
#define NN_INPUT_SETUP_INTERFACE       nrf_edgeai_input_setup_sliding_window 
#define NN_INPUT_FEED_INTERFACE        nrf_edgeai_input_feed_sliding_window_i16 
#define NN_PROCESS_FEATURES_INTERFACE  nrf_edgeai_process_features_dsp_i16_q16 
#define NN_RUN_INFERENCE_INTERFACE     nrf_edgeai_run_model_inference_q16 
#define NN_PROPAGATE_OUTPUTS_INTERFACE nrf_edgeai_output_dequantize_q16_f32 
#define NN_DECODE_OUTPUTS_INTERFACE    nrf_edgeai_output_decode_classification_f32 

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
    .model.meta.task                        = (nrf_edgeai_model_task_t)MODEL_TASK,
    .model.meta.uses_as_input.all           = MODEL_USES_AS_INPUT_MASK,

    .model.params.MODEL_PARAMS_TYPE = {
        .p_weights      = MODEL_WEIGHTS,
        .p_act_weights  = MODEL_NEURON_ACTIVATION_WEIGHTS,
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
    (sizeof(MODEL_WEIGHTS) + sizeof(MODEL_NEURONS_LINKS) + sizeof(MODEL_NEURON_EXTERNAL_LINKS_NUM) +
            sizeof(MODEL_NEURON_INTERNAL_LINKS_NUM) + sizeof(MODEL_NEURON_ACTIVATION_WEIGHTS) +
            sizeof(MODEL_NEURON_ACTIVATION_TYPE_MASK) +
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
