/* 2026-07-22T11:47:09.978318 */
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
#define EDGEAI_LAB_SOLUTION_ID_STR      "94826"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000202

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE                         f32

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE            NRF_EDGEAI_INPUT_F32

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM            6

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM       6

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE                  128

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT                 64

/** Number of subwindows in input feature window,
* the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
* if the window size is not divisible by the number of subwindows without a remainder,
* the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM                 0

#define INPUT_UNIQUE_SCALES_NUM (sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0])) 

/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = {
 -0.5236111, -1.6986110, -1.8375001, -5.2491651, -5.3322425, -2.8298333 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = {
 2.0041666, 1.6763890, 1.2694445, 4.3401976, 6.2402940, 3.4086280 };

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
#define MODEL_OUTPUTS_NUM          6

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 0
#define MODEL_USES_AS_INPUT_DSP_FEATURES 1
#define MODEL_USES_AS_INPUT_MASK ((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1)) 

#if MODEL_TYPE == __NRF_EDGEAI_MODEL_AXON 
#include <drivers/axon/nrf_axon_nn_infer.h>  
#include <axon/nrf_axon_platform.h> 
#include "nrf_edgeai_user_model_axon.h" 
#define P_MODEL_INSTANCE &model_axon_user_instance_94826
#else  // MODEL_TYPE == __NRF_EDGEAI_MODEL_NEUTON
#define P_MODEL_INSTANCE &model_neuton_user_instance_ 
#endif


#define NN_DECODED_OUTPUT_INIT                 \
.classif = {                                   \
   .predicted_class = 0,                       \
   .num_classes = MODEL_OUTPUTS_NUM,           \
}

//////////////////////////////////////////////////////////////////////////////
#define MODEL_NEURONS_NUM          29
#define MODEL_WEIGHTS_NUM          253
#define MODEL_PARAMS_TYPE          f32
#define MODEL_REORDERING           1

static const nrf_user_weight_t MODEL_WEIGHTS[] = {
 -0.1903200, -0.4936467, 0.8775259, -0.9276540, -0.8210544, 0.1566924,
 0.9999999, 0.2589524, 0.1130631, -0.3650900, -0.1860322, -0.0840083,
 0.9894494, 0.1934090, 0.4552860, -0.4269567, -0.1055891, 0.6026366,
 -0.2735631, 0.9685687, -0.2141258, 0.1550565, -0.3716605, -0.5763088,
 -0.2149096, 0.2899050, 0.4455861, 0.8616504, 0.5502923, -0.5097299,
 -0.9000666, -0.1542221, -0.2912815, -0.9648665, 0.8246544, -0.2081389,
 0.2250771, 0.5132959, -0.7218466, -0.0866837, -0.5347295, 0.0755734,
 0.7479229, 0.1790694, -0.6750893, 0.5000000, -0.1130290, 0.9578050,
 0.6855172, 1.0000000, 0.2955460, -0.1895104, -0.9834958, -0.2470027,
 -0.9293701, 1.0000000, 0.9991070, 0.9856858, -0.9623169, -0.1290675,
 -0.9999809, 0.2618815, -0.7636456, 0.1235597, 0.7892309, 0.2944833,
 0.5043772, 0.0634507, -0.8292988, 0.1849571, -0.2426721, 0.3418337,
 0.9116782, 0.6416851, -0.4726453, -0.7604296, 0.9865031, -0.6700844,
 -0.3792613, 0.4199981, -0.0612478, -0.0203314, -0.6441337, 0.7555809,
 -0.0587791, 0.9868509, -0.2132653, 1.0000000, 0.2274082, 0.5084311,
 -0.4135744, -0.9568039, -0.6318315, -0.2084224, -0.0918754, -0.7197607,
 0.4391619, 0.9980376, 0.9687500, -0.7770437, -0.6392882, 0.3191542,
 -0.2372872, 0.2544362, 0.0526080, 0.1875465, 0.8521351, -0.0433535,
 -0.8113517, 0.3294780, 0.1817797, 0.1515940, 0.8593750, 0.6508518,
 0.3377182, 0.2717701, -0.9991762, 0.9995489, -1.0000000, -0.7595105,
 -1.0000000, 0.7311385, -0.0507656, -0.4948756, 0.3064182, -0.2649277,
 -0.1377050, 0.1240413, 0.9082168, -0.3192359, 0.5532227, -0.6626999,
 0.3710206, -0.2784337, -0.4645940, 0.5856901, 0.0183528, 0.9999999,
 0.9999999, 0.0495940, -0.5156407, 0.0668985, -0.6006791, -0.8992596,
 -0.2387426, 0.1287951, 0.1381254, 0.3973781, -0.5625000, -0.8746126,
 0.9062500, -0.1359099, 0.5740710, -0.0953272, 0.8257816, -0.0309705,
 -0.2652214, -0.1664599, -0.1753554, 0.3778473, -0.7153343, 0.0317898,
 -0.9959636, -0.7926320, -0.4545671, 0.7046511, -1.0000000, 1.0000000,
 0.5773373, 0.1031532, 0.4575077, -1.0000000, -0.6606644, 0.8721524,
 0.4205949, 0.1419123, -0.4985770, -0.1910316, -0.1252862, -0.1877178,
 0.7167495, 1.0000000, 0.4977000, -0.0621930, 0.1945024, 0.9797535,
 0.1845367, -0.1264967, -0.4377451, -0.3659487, 0.8964074, -0.8501753,
 0.9472467, -0.2087959, 0.0549874, -0.2024370, -0.5307809, 1.0000000,
 0.1273953, -0.2035149, 0.1533521, 1.0000000, 0.4965805, 0.0776733,
 0.0784640, 0.5037233, 0.0730210, 0.4661945, -0.0597676, -0.3057669,
 -0.2701839, -0.1893857, -0.9960327, -0.8593323, 0.6309034, -0.0047185,
 -0.1482712, 0.4407218, 0.3281883, -0.1807493, -0.9935747, -0.3114540,
 -0.2341146, -0.3287650, 0.8473855, -1.0000000, 0.7941632, -1.0000000,
 -1.0000000, 0.8750000, -0.6625913, 0.1285024, 0.6005970, 0.2852658,
 0.9887557, 1.0000000, -0.1928521, -0.2363453, -0.1513716, 0.1363472,
 -0.1475402, 0.0421283, 0.1816425, -0.5000000, -0.9990234, 0.1698095,
 -0.9506761, -1.0000000, 0.4310319, 0.3750000, -1.0000000, -1.0000000,
 0.0335531 };

static const uint16_t MODEL_NEURONS_LINKS[] = {
 0, 3, 7, 19, 28, 32, 37, 39, 40, 42, 51, 53, 58, 7, 10, 15, 23, 29, 32,
 35, 36, 40, 46, 54, 58, 1, 5, 7, 10, 17, 18, 19, 26, 35, 37, 51, 56, 58,
 0, 10, 15, 18, 38, 41, 47, 49, 50, 58, 3, 0, 6, 15, 56, 58, 3, 3, 8, 11,
 12, 58, 5, 58, 0, 1, 2, 7, 37, 51, 58, 0, 1, 2, 3, 7, 2, 10, 15, 20, 27,
 31, 41, 48, 51, 58, 0, 7, 4, 9, 10, 18, 30, 36, 38, 41, 44, 52, 58, 2, 9,
 58, 3, 0, 3, 8, 18, 23, 24, 28, 58, 3, 0, 7, 15, 18, 24, 32, 58, 4, 12,
 58, 0, 2, 8, 7, 10, 53, 58, 0, 1, 0, 4, 36, 53, 57, 58, 0, 2, 7, 9, 14,
 40, 58, 0, 16, 19, 35, 58, 14, 16, 7, 33, 58, 2, 8, 14, 16, 17, 10, 17,
 42, 58, 0, 18, 23, 25, 58, 3, 11, 20, 58, 1, 8, 14, 13, 18, 39, 41, 50,
 58, 1, 2, 8, 9, 12, 14, 17, 0, 7, 21, 23, 28, 34, 54, 57, 58, 1, 2, 8, 14,
 15, 22, 23, 0, 10, 14, 21, 25, 54, 55, 58, 3, 14, 23, 24, 10, 12, 15, 20,
 34, 45, 46, 50, 51, 58, 1, 8, 22, 23, 24, 25, 58, 0, 15, 17, 22, 25, 15,
 16, 19, 22, 29, 43, 58, 0, 7, 14, 15, 16, 17, 18, 19, 27, 58 };

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = {
 0, 13, 25, 38, 49, 55, 61, 65, 74, 86, 99, 101, 110, 119, 123, 129, 140,
 144, 149, 157, 161, 169, 173, 186, 202, 214, 230, 236, 252 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = {
 13, 25, 38, 48, 54, 60, 62, 69, 84, 97, 100, 109, 117, 120, 127, 135, 142,
 147, 152, 161, 166, 170, 179, 195, 210, 224, 231, 243, 253 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 40.0000000, 39.9999847, 40.0000000, 37.2971611, 40.0000000, 35.0125008,
 40.0000000, 40.0000000, 39.9999847, 39.9999847, 39.9999847, 39.9999847,
 39.9999847, 39.9999847, 35.0125008, 35.0125008, 40.0000000, 40.0000000,
 40.0000000, 40.0000000, 40.0000000, 39.9999847, 39.9999847 };

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = {
 0xbf, 0xdb, 0xdf, 0xb };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = {
 28, 26, 10, 21, 13, 6 };

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
#define EXTRACTED_FEATURES_NUM  58

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
 0x88081ff00000000, 0xc80813f00000000, 0xc80812b00000000, 0x8081df00000000,
 0x8081f500000000, 0x80815a00000000 };

/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
static const nrf_user_input_t FEATURES_EXTRACTION_ARGUMENTS[] =
{ 2, 2, 2, 2, 2, 2 };

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
 -0.5236111, -0.3152778, 0.0097222, -0.3693250, 0.0014824, -4.7893863,
 -1.6992788, 0.0018537, 0.0031370, 0.0014326, -0.6649112, 2.5281048,
 -1.6986110, -0.4527778, 0.0124999, -0.5096898, 0.0014086, -4.8081074,
 0.0032757, 0.0013451, -0.3644370, -0.0554993, 2.2232282, -1.8375001,
 -0.9736112, -0.9881841, -5.9787197, 0.0049197, 0.0032371, -0.4502433,
 -0.0520378, 1.9629017, -5.2491651, -0.1475240, 0.0140499, -1.0537593,
 0.0023241, -1.6024172, 0.0028956, 0.0056878, 0.0029726, -0.5648001,
 -5.3322425, 0.0171042, 0.0026172, -5.8134027, -1.5610566, 0.0033963,
 0.0035362, 0.0030423, -0.6045297, 0.0042761, -0.4181586, 0.0028461,
 -1.6845907, 0.0036356, 0.0035834, -0.3466560 };

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
 1.0222223, 2.0041666, 2.1083336, 1.0559895, 0.5657309, 4.6481676,
 50.3409653, 0.6448533, 1.1541936, 0.2345363, 0.9828498, 4.0976095,
 0.9916667, 1.6763890, 2.1847222, 1.0050993, 0.3495566, 4.3726792,
 1.0051135, 0.1602690, 0.9858102, -0.0362953, 4.0332289, 0.9638889,
 1.2694445, 0.9772567, 3.5873871, 0.9881957, 0.2422135, 0.9904092,
 -0.0385608, 3.9729204, 0.1099557, 4.3401976, 7.5701475, 1.1429098,
 1.4309980, 62.8223419, 1.7271076, 2.0231926, 0.6021521, 1.0052491,
 -0.0006109, 11.5725365, 1.4824558, 6.4237428, 54.8517456, 1.8582290,
 1.8598132, 1.2419320, 0.9921500, 3.4086280, 0.2718135, 0.7971797,
 52.7983780, 1.0010600, 0.5042260, 0.9971103 };

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
    nrf_edgeai_feature_skew_kur_f32,
    nrf_edgeai_feature_std_f32,
    nrf_edgeai_feature_rms_f32,
    nrf_edgeai_feature_amdf_f32,
    nrf_edgeai_feature_autocorr_f32,
    nrf_edgeai_feature_lrp_f32
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
#define NN_PROCESS_FEATURES_INTERFACE  nrf_edgeai_process_features_dsp_f32_f32 
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
    .interfaces.init_inference      = NN_INIT_INFERENCE_INTERFACE,
    .interfaces.run_inference       = NN_RUN_INFERENCE_INTERFACE,
    .interfaces.propagate_outputs   = NN_PROPAGATE_OUTPUTS_INTERFACE,
    .interfaces.decode_outputs      = NN_DECODE_OUTPUTS_INTERFACE,
    ///
    .decoded_output = { NN_DECODED_OUTPUT_INIT },
};

//////////////////////////////////////////////////////////////////////////////

nrf_edgeai_t* nrf_edgeai_user_model_94826(void)
{
    return &nrf_edgeai_;
}

//////////////////////////////////////////////////////////////////////////////

uint32_t nrf_edgeai_user_model_size_94826(void)
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


