/* 2026-11-10T12:57:49.119862 */
/*
* Copyright (c) 2026 Nordic Semiconductor ASA
* SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
*/
#include "nrf_edgeai_user_model.h"
#include "nrf_edgeai_user_types.h"
#include <nrf_edgeai/nrf_edgeai_platform.h>
#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>
#include <assert.h>

//////////////////////////////////////////////////////////////////////////////
/* Nordic EdgeAI Lab Solution ID and Runtime Version */
#define EDGEAI_LAB_SOLUTION_ID_STR      "97286"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000103

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE                         f32

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE            NRF_EDGEAI_INPUT_F32

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM            1

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM       1

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE                  200

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT                 50

/** Number of subwindows in input feature window,
* the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
* if the window size is not divisible by the number of subwindows without a remainder,
* the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM                 0

#define INPUT_UNIQUE_SCALES_NUM (sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0])) 

/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = {
 0.3747816 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = {
 52.1913071 };

/** Defines which unique features from the input data will be used/collected,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USAGE_MASK NULL

/** Defines which unique input features is used for LAG features processing,
 *  one bit for one unique feature, starting from LSB
 */
#define INPUT_FEATURES_USED_FOR_LAGS_MASK NULL

//////////////////////////////////////////////////////////////////////////////
#define MODEL_TYPE                 __NRF_EDGEAI_MODEL_NEUTON
#define MODEL_TASK                 0
#define MODEL_OUTPUTS_NUM          4

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 0
#define MODEL_USES_AS_INPUT_DSP_FEATURES 1
#define MODEL_USES_AS_INPUT_MASK ((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1)) 

#if MODEL_TYPE == __NRF_EDGEAI_MODEL_AXON 
#include <drivers/axon/nrf_axon_nn_infer.h>  
#include <axon/nrf_axon_platform.h> 
#include "nrf_edgeai_user_model_axon.h" 
#define P_MODEL_INSTANCE &model_axon_user_instance_97286
#else  // MODEL_TYPE == __NRF_EDGEAI_MODEL_NEUTON
#define P_MODEL_INSTANCE &model_neuton_user_instance_ 
#endif


#define NN_DECODED_OUTPUT_INIT                 \
.classif = {                                   \
   .predicted_class = 0,                       \
   .num_classes = MODEL_OUTPUTS_NUM,           \
}

//////////////////////////////////////////////////////////////////////////////
#define MODEL_NEURONS_NUM          26
#define MODEL_WEIGHTS_NUM          201
#define MODEL_PARAMS_TYPE          f32
#define MODEL_REORDERING           1

static const nrf_user_weight_t MODEL_WEIGHTS[] = {
 0.2000098, -1.0000000, 0.6622760, -0.2136757, -1.0000000, -0.9833984,
 -1.0000000, 0.1375335, -0.1978491, 0.1729240, 0.2633843, 0.8302051,
 1.0000000, 0.0702587, -0.1267277, -0.1850121, 0.1148575, -0.1772040,
 0.4132064, 0.9445871, 0.1369895, -1.0000000, 0.6603172, 0.2301002,
 0.0601924, 0.0986780, -0.4464211, 0.7951254, -0.6375875, 0.4579740,
 0.1669727, 0.3734343, 0.0198321, 0.4984313, -0.8039871, -0.5575215,
 -0.2395691, 1.0000000, -0.5049550, 0.5490796, 0.0313686, 0.2243713,
 -0.2997028, 0.7761878, -0.9508200, 0.9995117, -0.5640710, -0.9735704,
 0.9104998, 1.0000000, -1.0000000, -0.7079268, -0.1638077, 0.4347750,
 0.3448969, 0.1112348, -0.9445871, 0.8675319, -0.9015400, 0.2172017,
 -0.8672931, -1.0000000, -0.1762692, 0.4543975, -0.9880486, 0.8578438,
 0.8791075, 1.0000000, -1.0000000, -0.7564134, -0.6017860, -0.0534351,
 0.9915187, 0.8747731, 0.9814466, 0.2231698, -0.9984472, -0.6508871,
 -0.2229337, 0.3386340, 0.8788776, 0.4623317, -0.6632159, 0.1031170,
 -0.5000000, -0.2865280, -1.0000000, -0.0490153, -0.1881108, -0.7857833,
 0.9648438, 0.0113325, -0.9575775, 0.1263060, 0.3236632, -0.1190677,
 -0.2242190, -1.0000000, -0.4592047, 0.3896144, -0.0734625, -1.0000000,
 0.8563861, -0.6833572, 0.6781881, 0.2437146, -0.4047306, 0.3245383,
 -0.3871256, -1.0000000, 0.1260611, 1.0000000, 0.3682663, 0.5963277,
 0.1412226, -0.5367121, -0.5036381, 0.2612851, 0.7724990, -0.7463289,
 -0.8125000, -1.0000000, -0.0033898, 0.6320638, -0.3484108, -0.0065273,
 -0.7291467, 0.0445766, -0.0219024, 1.0000000, -0.9375000, -0.7414134,
 0.4375000, -0.1591487, -0.9087307, -0.0748185, 0.9437231, 0.9999998,
 -1.0000000, -0.7724195, 0.1151963, 0.4371030, 1.0000000, 0.4508700,
 0.0104006, 0.9964968, -0.6131180, 1.0000000, -0.0976890, 0.2606229,
 -0.2221469, -0.8320473, 0.6443000, -0.3841732, -0.8976496, 0.9999999,
 -0.0892421, -0.7257429, -0.8873218, -0.3131665, -0.1977896, 1.0000000,
 1.0000000, 0.0086385, -0.6765062, 0.9062500, -1.0000000, 0.8750000,
 -1.0000000, 0.7357979, -1.0000000, 1.0000000, -1.0000000, 0.2420278,
 0.2640874, 0.0040016, -0.8013217, 1.0000000, -1.0000000, 0.0175334,
 -0.1029681, -0.0356652, 0.4081643, -0.4051731, -0.9921875, 0.7852661,
 -0.9687500, 0.6043551, 0.2596234, -0.2216685, -0.7121922, 0.7574303,
 -0.4765625, -1.0000000, 0.4321272, 1.0000000, -0.9205775, -0.6868666,
 -1.0000000, 1.0000000, -0.6316990 };

static const uint16_t MODEL_NEURONS_LINKS[] = {
 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 1, 2, 3, 4, 7, 9, 11, 0, 1, 0,
 3, 5, 7, 9, 10, 11, 1, 2, 2, 3, 4, 6, 7, 8, 9, 11, 2, 0, 1, 2, 3, 9, 11,
 0, 11, 0, 5, 11, 1, 2, 3, 0, 3, 5, 7, 10, 11, 3, 2, 4, 8, 10, 11, 3, 8,
 11, 2, 0, 1, 2, 3, 6, 9, 11, 1, 0, 5, 6, 11, 2, 4, 7, 0, 3, 4, 7, 8, 9,
 10, 11, 1, 2, 7, 12, 0, 10, 11, 7, 0, 4, 5, 10, 11, 2, 7, 10, 11, 12, 0,
 2, 4, 10, 11, 1, 2, 8, 12, 15, 0, 4, 9, 10, 11, 2, 7, 12, 13, 0, 9, 11, 1,
 2, 7, 16, 17, 0, 3, 11, 0, 1, 13, 15, 16, 17, 2, 8, 11, 1, 2, 13, 19, 2,
 3, 11, 7, 13, 16, 17, 19, 11, 1, 4, 15, 16, 17, 18, 19, 20, 21, 11, 1, 2,
 19, 20, 21, 2, 10, 11, 2, 19, 20, 21, 23, 2, 4, 6, 11, 2, 7, 10, 11, 12,
 13, 14, 23, 24, 11 };

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = {
 0, 13, 23, 32, 41, 48, 51, 55, 62, 69, 71, 79, 86, 98, 102, 112, 122, 131,
 139, 148, 155, 163, 173, 179, 187, 200 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = {
 12, 21, 30, 40, 47, 49, 52, 61, 67, 70, 78, 83, 94, 101, 107, 117, 127,
 134, 142, 151, 158, 164, 174, 182, 191, 201 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 40.0000000, 35.7043190, 39.9999847, 40.0000000, 40.0000000, 40.0000000,
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 40.0000000, 40.0000000 };

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = {
 0xbf, 0xfd, 0xbf, 0x1 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = {
 6, 22, 25, 9 };

/** Model neurons activations buffer */ 
static nrf_user_neuron_t model_neurons_[MODEL_NEURONS_NUM];

/** Neuton model instance */ 
static const nrf_edgeai_model_neuton_t model_neuton_user_instance_ = { 
   .meta.p_neuron_internal_links_num = MODEL_NEURON_INTERNAL_LINKS_NUM,
   .meta.p_neuron_external_links_num = MODEL_NEURON_EXTERNAL_LINKS_NUM,
   .meta.p_output_neurons_indices    = MODEL_OUTPUT_NEURONS_INDICES,
   .meta.p_neuron_links              = MODEL_NEURONS_LINKS,
   .meta.p_neuron_act_type_mask      = MODEL_NEURON_ACTIVATION_TYPE_MASK,
   .meta.outputs_num                 = MODEL_OUTPUTS_NUM,
   .meta.neurons_num                 = MODEL_NEURONS_NUM,
   .meta.weights_num                 = MODEL_WEIGHTS_NUM,
   /// 
   .params.MODEL_PARAMS_TYPE = {
       .p_weights      = MODEL_WEIGHTS,
       .p_act_weights  = MODEL_NEURON_ACTIVATION_WEIGHTS,
       .p_neurons      = model_neurons_,
   },
};

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
#define EXTRACTED_FEATURES_NUM  11

#define EXTRACTED_FEATURES_META_TYPE f32 

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

static const uint64_t FEATURES_EXTRACTION_MASK[] = {
 0x48839f00000000 };

/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
#define FEATURES_EXTRACTION_ARGUMENTS NULL

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
 0.3747816, 9.8530293, 0.0467634, 8.6121454, 0.0078692, 0.0097076,
 9.5338240, 0.0150754, 0.0096308, 0.1700000, 0.1684056 };

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
 9.9620094, 52.1913071, 47.5597458, 23.2338104, 8.7876244, 10.8126030,
 25.2842121, 0.5527638, 1.8115819, 0.9400000, 32.7708626 };

/** Memory allocation to store extracted features during DSP pipeline */
static uint8_t extracted_features_buffer_[EXTRACTED_FEATURES_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED;


/** Timedomain features processing context  */
#define P_TIMEDOMAIN_FEATURES_CTX  NULL
/** Timedomain features in feature extraction pipeline  */
static const nrf_edgeai_features_pipeline_func_f32_t timedomain_features_[] = {
    nrf_edgeai_feature_utility_tss_sum_f32,
    nrf_edgeai_feature_min_max_range_f32,
    nrf_edgeai_feature_mean_f32,
    nrf_edgeai_feature_mad_f32,
    nrf_edgeai_feature_std_f32,
    nrf_edgeai_feature_rms_f32,
    nrf_edgeai_feature_mcr_f32,
    nrf_edgeai_feature_amdf_f32,
    nrf_edgeai_feature_psom_f32,
    nrf_edgeai_feature_rmds_f32
 };

static const nrf_edgeai_features_pipeline_ctx_t timedomain_pipeline_ = {
    .functions_num     = sizeof(timedomain_features_) / sizeof(timedomain_features_[0]),
    .functions.p_void  = timedomain_features_,
    .p_ctx             = P_TIMEDOMAIN_FEATURES_CTX,
};
#define P_TIMEDOMAIN_PIPELINE &timedomain_pipeline_ 

#define P_FREQDOMAIN_PIPELINE NULL

#define P_CUSTOMDOMAIN_PIPELINE NULL

static nrf_edgeai_dsp_pipeline_t dsp_pipeline_ = { 
   .features = {  
       .p_masks = (const nrf_edgeai_features_mask_t*)FEATURES_EXTRACTION_MASK, 
       .buffer.p_void = extracted_features_buffer_, 
       .overall_num = EXTRACTED_FEATURES_NUM, 
       .masks_num = sizeof(FEATURES_EXTRACTION_MASK) / sizeof(FEATURES_EXTRACTION_MASK[0]), 

       .p_timedomain_pipeline = P_TIMEDOMAIN_PIPELINE, 
       .p_freqdomain_pipeline = P_FREQDOMAIN_PIPELINE, 
       .p_customdomain_pipeline = P_CUSTOMDOMAIN_PIPELINE, 

       .meta.EXTRACTED_FEATURES_META_TYPE = { 
           .p_min = EXTRACTED_FEATURES_SCALE_MIN, 
           .p_max = EXTRACTED_FEATURES_SCALE_MAX, 
       .p_arguments = FEATURES_EXTRACTION_ARGUMENTS, 
       },
   }, 
}; 

#define P_DSP_PIPELINE         &dsp_pipeline_ 


//////////////////////////////////////////////////////////////////////////////
#define NN_INPUT_INIT_INTERFACE        nrf_edgeai_input_init_sliding_window 
#define NN_INPUT_FEED_INTERFACE        nrf_edgeai_input_feed_sliding_window_f32 
#define NN_PROCESS_FEATURES_INTERFACE  nrf_edgeai_process_features_dsp_f32 
#define NN_SCALE_FEATURES_INTERFACE    nrf_edgeai_scale_features_dsp_f32_f32 
#define NN_INIT_INFERENCE_INTERFACE    nrf_edgeai_init_inference_neuton 
#define NN_RUN_INFERENCE_INTERFACE     nrf_edgeai_run_inference_neuton_f32 
#define NN_PROPAGATE_OUTPUTS_INTERFACE nrf_edgeai_output_propagate_neuton_f32 
#define NN_DECODE_OUTPUTS_INTERFACE    nrf_edgeai_output_decode_classification_f32 

//////////////////////////////////////////////////////////////////////////////

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
    .model.type                 = (nrf_edgeai_model_type_t)MODEL_TYPE,
    .model.task                 = (nrf_edgeai_model_task_t)MODEL_TASK,
    .model.instance.p_void      = P_MODEL_INSTANCE,
    .model.output.memory.p_void = model_outputs_,
    .model.output.num           = MODEL_OUTPUTS_NUM,
    .model.uses_as_input.all    = MODEL_USES_AS_INPUT_MASK,
    ///
    .interfaces.input_init          = NN_INPUT_INIT_INTERFACE,
    .interfaces.feed_inputs         = NN_INPUT_FEED_INTERFACE,
    .interfaces.process_features    = NN_PROCESS_FEATURES_INTERFACE,
    .interfaces.scale_features      = NN_SCALE_FEATURES_INTERFACE,
    .interfaces.init_inference      = NN_INIT_INFERENCE_INTERFACE,
    .interfaces.run_inference       = NN_RUN_INFERENCE_INTERFACE,
    .interfaces.propagate_outputs   = NN_PROPAGATE_OUTPUTS_INTERFACE,
    .interfaces.decode_outputs      = NN_DECODE_OUTPUTS_INTERFACE,
    ///
    .decoded_output = { NN_DECODED_OUTPUT_INIT },
};

//////////////////////////////////////////////////////////////////////////////

nrf_edgeai_t* nrf_edgeai_user_model_97286(void)
{
    return &nrf_edgeai_;
}

//////////////////////////////////////////////////////////////////////////////

uint32_t nrf_edgeai_user_model_size_97286(void)
{
    uint32_t model_size = 0;

#if MODEL_TYPE == __NRF_EDGEAI_MODEL_NEUTON
    model_size +=
        (sizeof(MODEL_WEIGHTS) + sizeof(MODEL_NEURONS_LINKS) +
         sizeof(MODEL_NEURON_EXTERNAL_LINKS_NUM) + sizeof(MODEL_NEURON_INTERNAL_LINKS_NUM) +
         sizeof(MODEL_NEURON_ACTIVATION_WEIGHTS) + sizeof(MODEL_NEURON_ACTIVATION_TYPE_MASK) +
         sizeof(MODEL_OUTPUT_NEURONS_INDICES));

#if MODEL_TASK == __NRF_EDGEAI_TASK_ANOMALY_DETECTION
    model_size += sizeof(MODEL_AVERAGE_EMBEDDING) + sizeof(MODEL_OUTPUT_SCALE_MIN) +
                  sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

#if MODEL_TASK == __NRF_EDGEAI_TASK_REGRESSION
    model_size += sizeof(MODEL_OUTPUT_SCALE_MIN) + sizeof(MODEL_OUTPUT_SCALE_MAX);
#endif

#elif MODEL_TYPE == __NRF_EDGEAI_MODEL_AXON
    const nrf_axon_nn_compiled_model_s* p_axon_model = P_MODEL_INSTANCE;

    model_size += sizeof(*p_axon_model);
    model_size += p_axon_model->model_const_size;
    model_size += p_axon_model->cmd_buffer_len * sizeof(NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE);

    if (p_axon_model->persistent_vars.buf_ptr != NULL)
    {
        model_size +=
            sizeof(nrf_axon_nn_model_persistent_var_s) * p_axon_model->persistent_vars.count;
    }

#endif

    return model_size;
}
