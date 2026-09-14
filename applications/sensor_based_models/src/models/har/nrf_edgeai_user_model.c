/* 2026-09-02T16:05:10.987447 */
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
#define EDGEAI_LAB_SOLUTION_ID_STR      "95877"
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
#define INPUT_WINDOW_SIZE                  50

/** Number of input feature samples on that the input window is shifted */
#define INPUT_WINDOW_SHIFT                 25

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
 2.0041666, 1.6763890, 1.2694445, 5.2488594, 6.2402940, 3.4086280 };

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
#define P_MODEL_INSTANCE &model_axon_user_instance_95877
#else  // MODEL_TYPE == __NRF_EDGEAI_MODEL_NEUTON
#define P_MODEL_INSTANCE &model_neuton_user_instance_ 
#endif


#define NN_DECODED_OUTPUT_INIT                 \
.classif = {                                   \
   .predicted_class = 0,                       \
   .num_classes = MODEL_OUTPUTS_NUM,           \
}

//////////////////////////////////////////////////////////////////////////////
#define MODEL_NEURONS_NUM          50
#define MODEL_WEIGHTS_NUM          449
#define MODEL_PARAMS_TYPE          f32
#define MODEL_REORDERING           1

static const nrf_user_weight_t MODEL_WEIGHTS[] = {
 0.0936801, -0.8177549, -0.6823770, 0.7140749, -0.2169975, 1.0000000,
 -0.3904504, 0.8866324, -0.4540330, 0.2479392, -0.0557776, 1.0000000,
 -0.5067291, 0.6422983, 0.6002772, -0.6142122, -0.4806648, 0.5844525,
 0.4503686, 0.0777227, -0.9932185, -0.2685513, 0.7652324, -0.8455333,
 0.2792519, 0.8092003, 0.9987643, 0.2131007, 0.2271067, 0.1857244,
 0.7471870, 0.6192518, 0.1893859, -0.6988577, -0.3191632, -0.0708779,
 0.2917711, -0.9854243, -0.0993824, -0.3199580, 0.6582220, -0.0944068,
 0.0166504, 0.0907128, 0.1018156, 0.1875081, -0.1051904, 0.8821113,
 1.0000000, 0.5244592, 0.9999131, 0.9999157, 1.0000000, -0.4946871,
 -0.8845613, -0.9999557, 0.4345316, 0.6096289, -0.9329510, -0.7971371,
 0.3663228, -0.8687483, 0.3852488, -0.1946564, 0.1194089, 0.5967410,
 1.0000000, -0.1404252, 0.0984178, 0.2292894, 0.2208572, -0.2098039,
 -0.0252109, 0.6003631, 0.3569838, 0.9999999, -0.0192380, -0.0448313,
 -0.1495349, 0.1840965, -0.9347821, 0.1750698, -0.9990379, -0.6884313,
 -0.2500001, -0.3467446, 0.7118441, -0.1570137, 0.8153731, 0.0887665,
 -0.9531258, 0.0752322, -0.1161492, -0.1608807, 0.4232024, 0.8189507,
 0.2135810, -0.4562374, 0.0688103, 0.4176509, -0.0931373, 0.6376159,
 0.5236740, 0.8028172, -0.2541025, -0.8758107, 0.0272034, -0.2500001,
 -0.5389400, -0.3181106, -0.9296227, -0.9988800, 0.6732646, 0.6646448,
 0.6314522, -0.3619547, -0.3453749, 0.5000000, -0.3380616, 0.7589630,
 0.0152289, -0.1423749, 0.2671396, 0.1234368, 0.0268752, 0.2459107,
 0.0502844, -0.0334419, 0.1204932, -0.4533811, 0.2011446, -0.2118325,
 -1.0000000, 0.2957281, -0.4437644, 1.0000000, -1.0000000, 0.4907942,
 0.2376563, -0.6593735, 0.5000000, 0.5108110, -0.5226139, -0.7916149,
 -0.4822945, 0.1161367, 0.4720622, 0.9534438, 0.6007923, 0.7498980,
 -1.0000000, -0.4534762, -0.2582236, -0.6748253, -0.9898150, -1.0000000,
 0.4350275, 0.6220138, -0.8431799, 0.2596714, -0.1052330, -0.1037983,
 0.0627344, 0.2864454, 0.1172870, -0.3308407, -0.1613858, -0.1441935,
 -0.1811909, 1.0000000, -0.5775535, 0.2968867, 0.4812627, 0.2693004,
 0.0011852, -0.7413720, 0.5619316, -0.1921285, -0.0028443, -0.6723831,
 0.0200974, -0.7902150, 0.9281609, -0.9999999, 0.0861118, -0.2500000,
 0.3469238, -0.3629335, 0.2806078, 0.6247093, 0.0934950, 0.4077095,
 -0.3128237, -0.9715010, 0.0156669, -0.4335970, -0.3223292, -0.0199807,
 0.6001804, -0.1959565, -0.2789573, -0.7276062, 0.7208531, 0.2100026,
 0.0369689, -0.6815065, 0.9788054, 0.2981174, 0.3899096, -0.3047544,
 0.4018548, 0.9972930, 0.5113375, -0.3702286, -0.2853427, 0.0521273,
 -0.0592524, -0.3905298, -0.9990348, 0.2198327, 0.2498312, 0.3234341,
 0.2039333, 0.3172503, -0.5683614, -1.0000000, 0.0840148, -0.5842464,
 0.2383014, -0.9961939, -0.8894812, -0.1458226, 0.7869707, 0.5115011,
 0.9899593, 0.4330936, -0.2831222, -0.8988721, 0.8370486, -1.0000000,
 -0.0526276, 0.0465729, 0.0199628, -0.0745893, -0.0359348, -0.4794858,
 -0.9280190, 0.8320788, -0.3772100, -0.1718266, 0.0045113, 0.4519958,
 -0.8834463, -0.8557248, 0.8526137, -0.3615581, 0.6977940, -0.1203816,
 0.0555978, 0.0154708, 0.2920060, -0.4810869, -0.3828452, -0.2867929,
 0.0131795, -0.7737501, -0.2459932, 0.6768399, 0.1368379, 0.2973498,
 0.1281497, 0.8752902, 0.0636915, -0.2456558, -0.6357195, 0.3588382,
 -0.9650449, -0.9375000, -0.4375000, 0.4302374, -0.1295311, -0.3544159,
 1.0000000, -0.0885025, 0.0159062, -1.0000000, -0.7218090, 0.0355004,
 -0.0265555, -0.8410708, -1.0000000, -1.0000000, -0.4919253, -0.5755669,
 0.0141509, -0.1635659, 0.0769747, -0.0750996, -0.1320888, 0.1773967,
 0.5977162, 0.0021984, -0.9890188, -0.5316089, 0.8750000, 0.0460676,
 0.9901613, -0.7893640, 0.0192782, -0.0586417, 0.1239192, 1.0000000,
 0.2936768, 0.6041507, 0.5966704, -1.0000000, -0.2483233, 0.1554443,
 -0.3555154, -0.1617930, -0.1238496, -0.0468989, -0.6835434, -0.5523461,
 -0.2953223, -0.9750977, 0.3648120, 0.5261367, -0.9852132, 0.1132240,
 0.2204346, -0.1098892, 0.1378747, -1.0000000, 0.3553309, -0.0473188,
 -0.6517096, 0.1426051, 0.3279369, -0.6807848, -0.5396912, -0.6321123,
 -0.1281410, -0.9716318, 0.7915716, 0.2044649, -0.0875437, 0.8309875,
 0.9878083, 0.4843917, 0.1633294, -0.1019879, -0.1767951, -0.7995731,
 0.1163039, 0.9999996, 1.0000000, 0.8592169, 0.3870286, 0.2308428,
 0.1059267, -0.1051306, 0.0023782, -0.0384021, -1.0000000, -0.7504700,
 0.6690148, -0.9901613, -0.7677014, -0.9943328, 0.6072891, -1.0000000,
 -0.6198436, -0.9125590, 1.0000000, -0.6064879, -1.0000000, -1.0000000,
 -0.9367195, -0.5840821, -0.5219395, -0.0649620, 0.9712906, 0.2044926,
 0.0012062, -1.0000000, 0.9345703, -1.0000000, 0.4563482, -1.0000000,
 1.0000000, -0.3862078, 0.5610338, -1.0000000, 0.0505634, 0.5834243,
 0.1225677, -1.0000000, -0.1266838, -0.0510787, 0.8055568, -1.0000000,
 0.9584489, 1.0000000, -1.0000000, 1.0000000, -0.4921875, -0.6246927,
 -0.2733803, 0.5968618, 0.7631534, 0.2578070, -0.0058465, 1.0000000,
 -0.5000000, 0.5000000, -0.5000000, -0.9980469, -1.0000000, 1.0000000,
 -0.9334714, -0.1796400, 1.0000000, 0.1193921, 0.9992960, -0.9682829,
 0.4150231, -0.9676876, 0.1321055, 0.0596996, 0.6356729, -0.1980528,
 0.1778000, -0.0239162, 0.8188909, 0.1663012, -0.2253642, -0.1174408,
 0.2156188, -0.4081790, -0.8307590, 0.9071217, -0.9062282, 0.4444010,
 -1.0000000, 0.2493419, -1.0000000, 0.5222617, 0.5275784 };

static const uint16_t MODEL_NEURONS_LINKS[] = {
 1, 7, 9, 11, 15, 19, 26, 39, 48, 51, 58, 59, 80, 82, 4, 7, 14, 20, 45, 46,
 51, 80, 82, 7, 12, 26, 51, 59, 82, 0, 0, 20, 36, 52, 82, 2, 3, 0, 15, 17,
 20, 25, 26, 33, 56, 60, 67, 82, 0, 2, 3, 8, 11, 17, 82, 5, 82, 0, 1, 2, 7,
 51, 82, 0, 1, 2, 7, 1, 4, 7, 9, 13, 15, 48, 51, 53, 64, 65, 76, 80, 82, 1,
 2, 7, 8, 51, 53, 80, 82, 0, 8, 1, 7, 14, 29, 51, 80, 82, 0, 8, 2, 7, 14,
 26, 39, 82, 0, 2, 14, 15, 24, 43, 62, 68, 82, 8, 10, 11, 0, 5, 7, 9, 10,
 11, 15, 33, 39, 49, 51, 54, 62, 68, 71, 72, 82, 8, 11, 11, 49, 51, 52, 66,
 82, 1, 2, 4, 8, 10, 1, 3, 15, 23, 27, 48, 63, 66, 79, 82, 1, 2, 3, 8, 15,
 10, 18, 52, 82, 1, 2, 10, 4, 82, 1, 2, 8, 15, 18, 32, 40, 45, 82, 2, 8,
 18, 18, 29, 38, 44, 73, 82, 2, 14, 16, 20, 29, 30, 38, 73, 78, 82, 2, 7,
 16, 20, 34, 69, 73, 81, 82, 2, 10, 13, 14, 32, 35, 47, 68, 70, 80, 82, 8,
 7, 26, 32, 33, 45, 51, 55, 82, 8, 10, 12, 14, 32, 53, 55, 70, 82, 3, 4,
 20, 31, 33, 58, 68, 82, 25, 3, 20, 26, 28, 32, 42, 82, 3, 25, 13, 17, 20,
 26, 33, 50, 52, 64, 68, 82, 3, 25, 31, 64, 82, 3, 12, 0, 8, 24, 26, 34,
 57, 82, 4, 25, 26, 24, 34, 55, 57, 82, 25, 27, 29, 30, 0, 9, 24, 32, 33,
 36, 38, 50, 82, 3, 25, 26, 27, 18, 24, 26, 32, 82, 0, 7, 8, 10, 22, 4, 8,
 82, 4, 25, 26, 0, 24, 26, 32, 65, 82, 10, 13, 25, 70, 82, 2, 33, 17, 31,
 32, 48, 67, 74, 75, 79, 80, 81, 82, 0, 10, 16, 22, 36, 37, 77, 82, 7, 8,
 11, 20, 3, 6, 19, 21, 34, 49, 51, 73, 82, 0, 7, 8, 9, 10, 11, 12, 13, 14,
 15, 33, 37, 38, 82, 1, 2, 22, 38, 82, 1, 16, 17, 18, 19, 20, 21, 36, 40,
 82, 8, 26, 33, 47, 82, 3, 25, 26, 27, 28, 29, 42, 82, 2, 22, 41, 46, 82,
 28, 34, 0, 82, 4, 30, 31, 32, 34, 45, 82, 9, 37, 45, 22, 59, 61, 68, 82,
 22, 3, 5, 12, 21, 62, 70, 82, 2, 22, 23, 24, 35, 44, 47, 48, 82 };

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = {
 0, 14, 23, 30, 37, 48, 56, 60, 67, 85, 91, 100, 107, 118, 137, 148, 163,
 170, 175, 184, 191, 201, 212, 221, 231, 240, 247, 256, 268, 273, 283, 292,
 305, 315, 321, 329, 334, 350, 357, 379, 384, 394, 395, 407, 410, 415, 423,
 427, 433, 448 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = {
 14, 23, 29, 35, 48, 55, 57, 63, 81, 89, 98, 106, 115, 135, 143, 158, 167,
 172, 181, 190, 200, 209, 220, 229, 238, 246, 254, 266, 271, 280, 288, 301,
 310, 318, 327, 332, 345, 353, 366, 380, 385, 395, 400, 408, 413, 417, 424,
 432, 440, 449 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 40.0000000, 40.0000000, 40.0000000, 40.0000000, 38.7531128, 38.7531128,
 38.7531128, 38.7531128, 38.7531128, 38.7531128, 39.9618454, 40.0000000,
 40.0000000, 39.9975586, 40.0000000, 40.0000000, 40.0000000, 40.0000000,
 39.9720688, 39.9720688, 39.9720688, 40.0000000, 39.9720688, 40.0000000,
 38.7531128, 40.0000000, 40.0000000, 40.0000000, 38.7531128, 38.7531128,
 40.0000000, 40.0000000, 40.0000000, 39.9720688, 39.9720688, 40.0000000,
 40.0000000, 40.0000000 };

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = {
 0xbf, 0xff, 0xff, 0xff, 0x7f, 0xb5, 0x1 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = {
 39, 41, 49, 43, 46, 6 };

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
#define EXTRACTED_FEATURES_NUM  82

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
 0x3c871ff00000000, 0xb04c5ef00000000, 0xbc4f31b00000000,
 0x14451df00000000, 0x340a00e00000000, 0xb4896c300000000 };

/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
static const nrf_user_input_t FEATURES_EXTRACTION_ARGUMENTS[] =
{ 4, 4, 2, 4, 4, 2, 4, 4, 4 };

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
 -0.5236111, -0.3763889, 0.0055556, -0.4062501, 0.0008957, -4.5840573,
 -1.7013857, 0.0013844, 0.0024056, 0.0027778, 0.0041665, 0.0022222,
 0.1200000, 0.0112835, -0.8016894, 0.1218375, 0.8336315, -1.6986110,
 -0.5416667, 0.0069444, -0.5644445, -4.7890816, -1.8239065, 0.0013310,
 0.0027916, 0.0000000, 0.0026111, 0.0008787, 0.0000000, 0.0797954,
 0.9275711, 2.1590328, -1.8375001, -0.9791667, -0.9919998, 0.0022222,
 0.0037060, 0.0204082, 0.0055555, 0.0086806, 0.0030556, 0.0020975,
 0.0000000, 0.0229902, -0.7840536, 0.0961144, 0.7906172, 1.8833603,
 -5.2491651, -1.6481144, 0.0097738, -2.7401032, 0.0017446, -1.8545012,
 0.0021641, 0.0036396, 0.0032834, 0.0028894, 0.0000000, 0.0204503,
 0.0688838, -0.2507602, 0.0106901, -0.8553334, 0.0078649, 0.0025868,
 0.0224695, 0.1099132, 0.8398871, -2.8298333, -0.2809980, -1.7507215,
 0.0027691, 0.0204082, 0.0000000, 0.0045815, 0.0027988, 0.1000000,
 0.0253178, 0.0958749, 0.8918951, 0.5152078 };

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
 1.0236112, 2.0041666, 2.1083336, 1.1373891, 0.6342890, 5.6919527,
 34.8850784, 0.6991962, 1.2350292, 2.0031252, 1.2697918, 1.1373891,
 0.8600000, 2.7691011, 0.9641060, 1.6810473, 9.2080040, 0.9972222,
 1.6763890, 2.1847222, 1.0091391, 5.1652327, 30.4569435, 0.4064863,
 1.0091954, 0.3469388, 1.0091391, 0.2134354, 1.0000000, 1.7539893,
 11.8808641, 3.9276564, 0.9680556, 1.2694445, 0.9851666, 0.3609412,
 0.9920926, 0.6734694, 1.5687500, 1.5579861, 0.9919998, 0.3239796,
 1.0000000, 3.2757399, 0.9887866, 1.5810714, 12.3749561, 3.8881049,
 1.0057896, 5.2488594, 6.8032060, 2.8234251, 1.8713427, 38.1426201,
 2.0802433, 3.0200841, 6.2992420, 2.8234251, 1.0000000, 7.4873829,
 1.7516551, 6.2402940, 11.5725365, 0.7242113, 7.7460771, 1.6619147,
 16.0143852, 1.6589170, 9.5249214, 0.1624902, 3.4086280, 34.9707603,
 1.2027005, 0.7551020, 0.6938776, 3.6951239, 0.6429856, 0.9200000,
 6.0622654, 1.7459579, 11.5315018, 4.2061582 };

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
    nrf_edgeai_feature_mcr_f32,
    nrf_edgeai_feature_zcr_f32,
    nrf_edgeai_feature_p2p_lf_hf_f32,
    nrf_edgeai_feature_absmean_f32,
    nrf_edgeai_feature_amdf_f32,
    nrf_edgeai_feature_psoz_f32,
    nrf_edgeai_feature_psom_f32,
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

nrf_edgeai_t* nrf_edgeai_user_model_95877(void)
{
    return &nrf_edgeai_;
}

//////////////////////////////////////////////////////////////////////////////

uint32_t nrf_edgeai_user_model_size_95877(void)
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


