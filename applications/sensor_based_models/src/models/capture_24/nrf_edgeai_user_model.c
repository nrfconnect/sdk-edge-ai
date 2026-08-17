/* 2026-07-31T15:03:58.380004 */
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
#define EDGEAI_LAB_SOLUTION_ID_STR      "95137"
#define EDGEAI_RUNTIME_VERSION_COMBINED 0x00000202

//////////////////////////////////////////////////////////////////////////////
#define INPUT_TYPE                         f32

/** User input features type */
#define INPUT_FEATURE_DATA_TYPE            NRF_EDGEAI_INPUT_F32

/** Number of unique features in the original input sample */
#define INPUT_UNIQ_FEATURES_NUM            3

/** Number of unique features actually used by NN from the original input sample */
#define INPUT_UNIQ_FEATURES_USED_NUM       3

/** Number of input feature samples that should be collected in the input window
 *  feature_sample = 1 * INPUT_UNIQ_FEATURES_NUM
 */
#define INPUT_WINDOW_SIZE                  200

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT                 200

/** Number of subwindows in input feature window,
* the SUBWINDOW_SIZE = INPUT_WINDOW_SIZE / INPUT_SUBWINDOW_NUM
* if the window size is not divisible by the number of subwindows without a remainder,
* the remainder is added to the last subwindow size */
#define INPUT_SUBWINDOW_NUM                 0

#define INPUT_UNIQUE_SCALES_NUM (sizeof(INPUT_FEATURES_SCALE_MIN) / sizeof(INPUT_FEATURES_SCALE_MIN[0])) 

/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = {
 -8.1049004, -8.2540998, -8.2678003 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = {
 8.1471996, 8.1114998, 8.4202003 };

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
#define P_MODEL_INSTANCE &model_axon_user_instance_95137
#else  // MODEL_TYPE == __NRF_EDGEAI_MODEL_NEUTON
#define P_MODEL_INSTANCE &model_neuton_user_instance_ 
#endif


#define NN_DECODED_OUTPUT_INIT                 \
.classif = {                                   \
   .predicted_class = 0,                       \
   .num_classes = MODEL_OUTPUTS_NUM,           \
}

//////////////////////////////////////////////////////////////////////////////
#define MODEL_NEURONS_NUM          43
#define MODEL_WEIGHTS_NUM          381
#define MODEL_PARAMS_TYPE          f32
#define MODEL_REORDERING           1

static const nrf_user_weight_t MODEL_WEIGHTS[] = {
 -0.1611632, -0.3342972, -0.2608822, 1.0000000, 1.0000000, 0.1901473,
 -0.0685606, -0.0597411, -0.0487972, 0.5000000, -1.0000000, -1.0000000,
 0.2874203, -1.0000000, -1.0000000, -0.1252826, 0.0928888, 0.5544140,
 -1.0000000, -0.0110486, 0.5818918, 0.0503003, 0.3639247, -0.5841445,
 -0.7590132, -0.2398054, -0.2651040, 1.0000000, 0.0572528, 0.1843203,
 -0.0660797, 0.5000000, 0.0641039, 0.5441412, -0.9969906, 0.9999999,
 0.8872414, 0.9482234, -1.0000000, -1.0000000, -1.0000000, -1.0000000,
 -1.0000000, -1.0000000, -1.0000000, -1.0000000, -1.0000000, -1.0000000,
 -1.0000000, -1.0000000, -0.3845922, -0.8338559, -0.0849799, 0.5215759,
 -1.0000000, -1.0000000, -0.1880659, -1.0000000, -0.9817235, -0.1515682,
 0.0220852, 0.0687868, 0.2232437, -0.9660308, 0.0744053, -0.1426467,
 0.3362791, -0.4806198, 0.1691287, -0.8980811, -0.1536622, 0.4023117,
 0.3031557, -1.0000000, -1.0000000, -1.0000000, -0.9738313, 0.0476973,
 -0.6157411, 1.0000000, -0.0929588, -0.9843750, 1.0000000, -0.0901226,
 -0.2357669, -0.2513500, 0.2376047, -0.9950118, -0.1480088, 0.5643581,
 -0.6250000, 0.3492327, -0.4372988, 0.0105908, 0.0025660, 0.3788022,
 -0.3609444, -0.0591851, 0.8978587, -0.1311377, 0.0204450, 0.1238532,
 0.5152858, -1.0000000, -1.0000000, -1.0000000, -1.0000000, -0.8673980,
 -1.0000000, -0.7833039, -1.0000000, 0.0331911, -0.1105879, -0.1746573,
 0.7060729, 1.0000000, -0.1509957, 0.0304916, 0.8749949, -1.0000000,
 -0.1493578, 0.0063450, -0.1001429, -0.1000287, -0.0626308, 0.6004910,
 -1.0000000, -1.0000000, -0.7553753, 0.3512939, -0.3461360, -0.5160073,
 0.0612027, 0.1262793, 0.5349551, 0.3707966, 0.3092058, -0.9664876,
 0.1444525, 0.9775602, 0.9981358, -0.7418357, -0.0217818, -0.7812448,
 0.8401650, 0.0811506, 0.6239697, 0.4608092, -0.1314430, 0.0645793,
 -0.6036447, -0.5988095, -0.9290581, 0.5000000, -0.5000000, 1.0000000,
 -0.4794853, 1.0000000, -0.7687179, -0.7983205, 0.9795665, -0.0999933,
 0.4116153, -0.8168205, 0.0630413, 0.1514156, -0.1514292, 0.4507428,
 -1.0000000, -1.0000000, 0.4539776, -1.0000000, 0.0350965, 0.4162400,
 0.0054370, -0.4149185, 0.2373646, -0.9782207, 0.2629276, 0.0328077,
 -0.2947957, -0.9131294, 0.1233471, -1.0000000, 0.0329940, -0.2339283,
 0.5806603, -1.0000000, 0.1957737, 0.9997559, 0.9999999, 0.2500000,
 0.1130672, -0.4734191, 0.8147424, -0.0560157, -0.0520050, 0.9999999,
 -0.3433672, 0.2500003, 0.8018504, -0.2693412, 0.0056969, -0.8736399,
 0.9921875, -0.7066383, -1.0000000, 0.0124572, -1.0000000, 0.0082414,
 0.2229736, 1.0000000, 0.0016096, 0.0008977, 0.0014532, 1.0000000,
 -1.0000000, 0.4284704, -1.0000000, 0.2664106, -0.2595124, 0.0064790,
 1.0000000, 0.1691731, 1.0000000, -0.0837281, -0.2898222, -0.9588820,
 0.1394248, 0.0153752, -0.6895598, 0.0165284, 0.2851619, 0.9990935,
 0.8450170, -0.4778472, 0.9914788, -0.9717391, 1.0000000, 0.5286721,
 -0.8281250, 0.0138370, -0.1338385, -0.8503200, 1.0000000, 0.1959521,
 0.0177862, 0.2946245, 0.1465618, -0.9687500, -0.3322405, 0.0030148,
 -0.9957645, -0.3247248, 0.2740581, 0.4974504, -0.1983956, -0.8750000,
 0.9843750, -0.9999999, 0.0600555, 0.2558410, -1.0000000, 0.0081017,
 -0.0864130, -0.4567834, -1.0000000, -1.0000000, -0.1053647, 0.0236083,
 -0.2609949, 0.1558685, 0.1253650, 0.0156172, -0.0451095, 0.8819444,
 -0.8750000, 0.5000000, -0.5000000, 0.1158376, 0.2472030, -0.0893134,
 0.2558614, -0.1835963, -1.0000000, 0.9999999, 0.9990616, -0.8745764,
 -0.1673021, -0.9750705, -0.1888413, -0.4281760, -0.7713402, 1.0000000,
 0.1131727, 0.3460449, -0.7525054, 0.7499207, -0.7578117, -0.4725440,
 -0.3732199, 0.5312499, -0.9374962, -0.2736053, -0.0755723, -1.0000000,
 1.0000000, 0.5026227, 0.8231543, -0.1987006, -0.4148569, -1.0000000,
 -0.3048773, -0.4171889, 0.6607864, -1.0000000, 1.0000000, 0.0003583,
 1.0000000, -0.8980545, 1.0000000, -0.1572636, -0.1049168, 0.8750000,
 -1.0000000, 0.8484530, -0.0458551, -0.7126328, -0.5000000, -1.0000000,
 1.0000000, 1.0000000, 0.4920368, 0.4273008, -0.0241878, -0.5196841,
 -1.0000000, -1.0000000, -0.7111945, -0.8205462, 0.5948189, -0.1868402,
 -0.1515597, -0.3625934, 0.8090625, 0.4565305, -1.0000000, 0.6417849,
 -0.5383948, -1.0000000, -0.7181002, -1.0000000, 1.0000000, -0.8239206,
 0.0708706, -0.7500000, 1.0000000, 0.8671910, 0.3170319, -0.2500001,
 -0.2382812, 0.8750000, 0.2369050, -0.2500000, -0.3246892, 1.0000000,
 -0.5000000, 0.5265334, -0.0364988, 0.5046946, -0.8437500, 1.0000000,
 1.0000000, -1.0000000, 1.0000000, 0.5788451, -0.8750000, -0.9687500,
 0.4062500, -1.0000000, -0.5114765 };

static const uint16_t MODEL_NEURONS_LINKS[] = {
 14, 17, 18, 22, 26, 28, 29, 36, 42, 6, 7, 22, 32, 36, 38, 39, 40, 42, 0,
 2, 3, 12, 15, 16, 17, 18, 20, 22, 24, 27, 28, 36, 37, 42, 0, 17, 42, 1, 1,
 2, 3, 7, 10, 16, 17, 20, 26, 31, 34, 36, 42, 1, 2, 3, 1, 3, 18, 20, 22,
 27, 28, 40, 42, 1, 5, 14, 18, 20, 22, 26, 28, 42, 5, 2, 7, 22, 32, 42, 1,
 4, 16, 18, 22, 25, 27, 28, 29, 35, 40, 42, 8, 16, 24, 29, 42, 0, 3, 8, 17,
 33, 36, 42, 1, 5, 7, 15, 20, 21, 22, 31, 34, 39, 40, 42, 8, 9, 9, 14, 18,
 19, 33, 37, 40, 41, 42, 3, 8, 9, 7, 14, 20, 23, 24, 28, 33, 40, 41, 42, 2,
 6, 8, 5, 12, 14, 16, 42, 3, 12, 13, 14, 42, 1, 11, 12, 5, 7, 15, 22, 31,
 34, 35, 39, 42, 1, 15, 22, 39, 42, 0, 12, 16, 2, 14, 22, 42, 1, 4, 16, 0,
 14, 15, 17, 18, 22, 33, 34, 35, 36, 42, 4, 12, 16, 14, 16, 22, 27, 42, 13,
 13, 22, 29, 42, 1, 12, 16, 20, 1, 33, 34, 42, 19, 20, 13, 27, 28, 37, 42,
 12, 16, 1, 33, 42, 0, 0, 24, 27, 37, 42, 0, 0, 27, 39, 42, 0, 14, 24, 27,
 42, 14, 16, 1, 33, 42, 0, 12, 4, 8, 16, 18, 21, 22, 25, 26, 27, 42, 0, 29,
 16, 17, 22, 24, 25, 26, 42, 0, 5, 25, 30, 3, 16, 17, 33, 40, 41, 42, 3,
 23, 25, 31, 4, 10, 26, 28, 36, 42, 0, 8, 11, 17, 22, 28, 29, 30, 33, 37,
 41, 42, 0, 10, 29, 30, 31, 32, 33, 42, 0, 11, 8, 14, 39, 42, 0, 16, 20,
 30, 32, 14, 27, 29, 42, 31, 32, 11, 16, 18, 27, 28, 42, 0, 20, 32, 33, 11,
 28, 36, 42, 20, 32, 11, 12, 14, 16, 18, 38, 42, 2, 4, 5, 6, 7, 8, 9, 18,
 20, 21, 23, 25, 26, 27, 35, 36, 37, 38, 39, 42, 1, 20, 1, 29, 33, 42, 1,
 11, 16, 17, 19, 22, 24, 28, 41, 42 };

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = {
 0, 9, 19, 35, 38, 54, 65, 73, 80, 90, 97, 103, 116, 128, 141, 150, 154,
 164, 171, 178, 192, 198, 206, 212, 219, 223, 229, 234, 240, 245, 257, 268,
 279, 286, 304, 307, 316, 322, 332, 338, 364, 367, 380 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = {
 9, 18, 34, 37, 51, 63, 72, 78, 90, 95, 102, 114, 125, 138, 146, 151, 163,
 168, 175, 189, 197, 202, 210, 217, 222, 228, 233, 238, 243, 255, 264, 275,
 285, 297, 305, 311, 320, 328, 336, 345, 365, 371, 381 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 20.2387390, 20.2387390,
 20.2387390, 20.2387390, 25.1790543, 25.1790543, 28.7781334, 40.0000000,
 32.5187531, 35.6155777, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 25.1790543, 40.0000000, 25.1790543, 25.1790543, 40.0000000, 25.1790543,
 40.0000000, 25.1790543, 25.1790543, 25.1790543, 40.0000000, 28.7781334,
 39.6047821, 39.6047821, 39.6047821, 39.6047821, 39.6047821, 25.1790543,
 25.1790543, 25.1790543, 25.1790543, 25.1790543, 25.1790543, 40.0000000,
 40.0000000 };

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = {
 0xff, 0x7f, 0xff, 0xff, 0xfb, 0x2 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = {
 34, 42, 40, 15 };

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
#define EXTRACTED_FEATURES_NUM  42

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
 0x9e4b19600000000, 0x9e5b59600000000, 0x9c4b19200000000 };

/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
static const nrf_user_input_t FEATURES_EXTRACTION_ARGUMENTS[] =
{ 4, 4, 2, 4, 4, 2, 2, 4, 4, 2 };

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
 -1.0183001, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, -0.8615626,
 0.0000000, 1.4873841, -0.9996000, 0.0000000, 0.0000000, 0.0000000,
 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000,
 0.0000000, 0.0000000, 0.0000000, -0.8924621, 0.0000000, 0.9844793,
 -1.0781000, 0.0000000, 0.0000000, 0.0019106, 0.0000000, 0.0000000,
 0.0000000, 0.0000000, 0.0000000, -0.9002017, 0.0000000, 1.3906903 };

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
 8.1471996, 15.7248001, 1.8211473, 2.3113379, 2.6964154, 7.3413992,
 15.7754250, 1.6069405, 1.0000000, 14.1421356, 39.4546165, 0.9980295,
 1.9659959, 4.3880529, 8.1114998, 16.2318001, 1.8470938, 2.3789303,
 2.8810284, 0.7236181, 7.0701499, 16.3672009, 2.7677236, 0.3618090,
 1.0000000, 13.3551674, 51.6602135, 1.0021451, 1.9267541, 4.8751907,
 8.4202003, 1.1286107, 1.6924492, 1.7719772, 7.2662249, 13.4408741,
 1.5313592, 1.0000000, 31.6801052, 1.0005215, 1.9623281, 4.4023056 };

/** Memory allocation to store extracted features during DSP pipeline */
static uint8_t extracted_features_buffer_[EXTRACTED_FEATURES_BUFFER_SIZE_BYTES] __NRF_EDGEAI_ALIGNED;


/** Timedomain features processing context  */
#define P_TIMEDOMAIN_FEATURES_CTX  NULL
/** Timedomain features in feature extraction pipeline  */
static const nrf_edgeai_features_pipeline_func_f32_t timedomain_features_[] = {
    nrf_edgeai_feature_utility_tss_sum_f32,
    nrf_edgeai_feature_min_max_range_f32,
    nrf_edgeai_feature_mad_f32,
    nrf_edgeai_feature_std_f32,
    nrf_edgeai_feature_rms_f32,
    nrf_edgeai_feature_zcr_f32,
    nrf_edgeai_feature_p2p_lf_hf_f32,
    nrf_edgeai_feature_amdf_f32,
    nrf_edgeai_feature_pscr_f32,
    nrf_edgeai_feature_psoz_f32,
    nrf_edgeai_feature_crest_f32,
    nrf_edgeai_feature_rmds_f32,
    nrf_edgeai_feature_autocorr_f32,
    nrf_edgeai_feature_hjorth_f32,
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
#define NN_INPUT_INIT_INTERFACE        nrf_edgeai_input_init_discrete_window 
#define NN_INPUT_FEED_INTERFACE        nrf_edgeai_input_feed_discrete_window_f32 
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

nrf_edgeai_t* nrf_edgeai_user_model_95137(void)
{
    return &nrf_edgeai_;
}

//////////////////////////////////////////////////////////////////////////////

uint32_t nrf_edgeai_user_model_size_95137(void)
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


