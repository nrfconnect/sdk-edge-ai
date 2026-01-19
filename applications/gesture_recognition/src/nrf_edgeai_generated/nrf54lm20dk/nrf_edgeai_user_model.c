/* 2026-01-23T11:10:10.318845 */
/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
#include "nrf_edgeai_user_model.h"
#include "nrf_edgeai_user_types.h"
#include <nrf_edgeai/nrf_edgeai_platform.h>
#include <nrf_edgeai/rt/private/nrf_edgeai_interfaces.h>

//////////////////////////////////////////////////////////////////////////////
/* Nordic EdgeAI Lab Solution ID and Runtime Version */
#define EDGEAI_LAB_SOLUTION_ID_STR      "91277"
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
#define MODEL_NEURONS_NUM          60
#define MODEL_WEIGHTS_NUM          615
#define MODEL_OUTPUTS_NUM          8
#define MODEL_TASK                 0
#define MODEL_PARAMS_TYPE          q16
#define MODEL_REORDERING           1

#define MODEL_USES_AS_INPUT_INPUT_FEATURES 0
#define MODEL_USES_AS_INPUT_DSP_FEATURES 1
#define MODEL_USES_AS_INPUT_MASK ((MODEL_USES_AS_INPUT_INPUT_FEATURES << 0) | (MODEL_USES_AS_INPUT_DSP_FEATURES << 1)) 

//////////////////////////////////////////////////////////////////////////////
/** Defines input(also used for LAG) features MIN scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MIN[] = {
 -32748, -32730, -32765, -17453, -17453, -17453 };

/** Defines input(also used for LAG) features MAX scaling factor
 */
static const nrf_user_input_t INPUT_FEATURES_SCALE_MAX[] = {
 32754, 32726, 32765, 16426, 17453, 17453 };

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
#define EXTRACTED_FEATURES_NUM  84

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

static const uint64_t FEATURES_EXTRACTION_MASK[] = {
 0x5dc79b00000000, 0x5cc79b00000000, 0x5cc39b00000000, 0x55c78b00000000,
 0x5dc79b00000000, 0x5cc79b00000000 };

/** Defines arguments used while feature extraction
 */

/** Defines arguments used while feature extraction
 */
static const nrf_user_input_t FEATURES_EXTRACTION_ARGUMENTS[] =
{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };

/** Defines extracted features MIN scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MIN[] = {
 -32748, -7267, -9425, 40, 56, 70, 10, 0, 59, 20, 0, 0, 121, 0, 256,
 -32730, -7666, -13897, 29, 38, 50, 10, 0, 40, 18, 0, 171, 0, 236, -32765,
 -284, -7485, 47, 91, 529, 10, 434, 27, 0, 131, 0, 343, -17453, 10, -2865,
 6, 7, 10, 0, 5, 2, 0, 10, 0, 27, -17453, -3494, -8439, 4, 6, 6, 10, 0, 4,
 2, 0, 0, 161, 0, 36, -17453, -483, -4905, 5, 8, 8, 10, 0, 5, 3, 0, 131, 0,
 40 };

/** Defines extracted features MAX scaling factor
 */
static const nrf_user_feature_t EXTRACTED_FEATURES_SCALE_MAX[] = {
 4885, 32754, 11374, 15412, 18479, 19097, 326, 306, 16738, 8404, 173, 1000,
 838, 313, 65435, 6514, 32726, 9336, 23603, 24653, 24755, 397, 377, 23726,
 18015, 1000, 858, 323, 64737, 9803, 32765, 13463, 18790, 21102, 21262,
 387, 18878, 14207, 1000, 878, 282, 65191, 597, 16426, 6722, 8456, 8932,
 244, 255, 8321, 1701, 183, 1000, 323, 31513, 407, 17453, 3480, 10450,
 11906, 11950, 346, 346, 10639, 3293, 244, 1000, 888, 303, 47527, 544,
 17453, 3967, 10020, 11041, 11054, 244, 255, 10048, 4000, 1000, 868, 313,
 47815 };

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
    nrf_edgeai_feature_pscr_i16,
    nrf_edgeai_feature_psoz_i16,
    nrf_edgeai_feature_psom_i16,
    nrf_edgeai_feature_psos_i16,
    nrf_edgeai_feature_rmds_i16
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
       .p_masks = (const nrf_edgeai_features_mask_t*)FEATURES_EXTRACTION_MASK, 
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
static const nrf_user_weight_t MODEL_WEIGHTS[] = {
 16383, 27246, 32760, 20549, 29684, -31731, -16383, 14007, 32760, 32765,
 32758, 32766, -26658, -32766, -8243, 32750, 32767, -20360, -16323, -9320,
 -15278, 17371, 9040, 25255, -6832, 12277, 14361, 30891, -5700, -27069,
 -9321, 5143, 3275, 12665, -30638, 24531, 9915, 10399, 10956, 32761, 7195,
 4118, -24394, 12334, -3804, 3112, -8757, 32756, -7719, 19886, 17338, 9396,
 6055, 8065, -18397, -26075, -10665, 21402, -28355, -32768, 30064, -6317,
 9451, 32766, 32743, 21310, 11374, 7035, 18232, -28651, 32765, -32766,
 -522, 5595, 26408, -31419, -28671, 27672, 2883, 10654, -8261, -32766,
 -32766, -12539, -5713, 8960, -3753, 10590, 15793, -18143, -17830, 2052,
 3967, -8420, 24562, 8765, -6091, -2561, -26035, 2702, -20248, 847, 20071,
 -2539, 11885, 32747, 27709, 5859, 19606, -14029, -8870, -29389, 3484,
 -2021, -3283, 28055, -8231, -20746, 7094, -13260, -18320, 11826, 32545,
 8425, 10649, -3346, -5463, 32535, -5636, 190, 26032, -16793, -214, -21793,
 -32760, 32764, -12981, -12199, -13354, 25047, 535, 23704, -26818, -18583,
 -1286, -1183, -32766, -6577, -32001, 9841, -24576, 18517, 6168, -38,
 -32766, 2988, -2877, -3072, 5427, 24565, 25792, -16526, -5094, 23285,
 -1680, -32060, -32560, -30632, 2560, 32761, -5628, -10665, 8208, -22924,
 -32765, 32766, 4072, 32764, -18098, -32768, -32768, 27636, 22627, -9971,
 32763, 5174, 20345, 2404, 28259, -16125, 5529, -1311, -5435, 4930, -20150,
 -32766, -2864, 9006, 19587, 901, 10636, 10532, 527, 3855, 744, -17749,
 21825, -32766, 13497, 9465, -31853, -8076, -18198, -21903, -6004, 3319,
 16383, 26392, 16480, 3268, 2852, -24857, 26570, -8307, -32766, -15725,
 -6496, 3289, -32766, 5489, -11548, -30665, 4044, 4836, -32766, -4817,
 15313, -22201, -22222, -14798, -32766, 27702, 6650, 10693, -32766, -1510,
 -3043, -20447, 6419, -32766, 11728, -30957, -769, 32765, 32760, -23621,
 -28021, -13164, 3570, -32766, 10679, -16887, 21081, 13778, -3773, 404,
 32765, 4216, 31740, 19258, -7030, -8716, 30138, -32125, -3428, 22999, 202,
 -2778, -12656, -32696, -32766, 17103, -10706, 8228, 7490, 19385, -9803,
 -32766, -5376, -8572, 19468, -21449, 24029, 28758, 21152, -14040, 32764,
 -16383, 32766, -20090, 32764, 359, 4638, -29576, 6731, -16387, -12293,
 1658, 13650, -32766, 8148, 14821, -3892, -11332, 32767, 32756, -2156,
 32766, -31743, -262, 733, 10698, 31973, -15096, -3230, -13202, 15992,
 1515, -20285, 32766, 24732, -17837, -32768, -32766, -14871, -15443, 32767,
 15943, -32766, -32766, -28353, -32766, 8548, -19214, 9005, -20002, 27859,
 8515, 16871, 27512, -10124, 21781, -30566, -32766, 31929, 31140, -17606,
 15881, 8675, -24923, 7894, 10614, 26356, -11192, 23009, 1279, -3676,
 11817, -10249, 32650, 599, -22902, 15104, 20208, -20741, -18772, 4592,
 25582, -30536, 32766, -6967, 8732, 32766, -8139, -32403, -10868, 11186,
 32664, 2535, -9694, -26697, 872, 30708, 32185, 10528, 23547, 11240,
 -20364, 25448, 27132, -6700, -31545, 9029, -3782, -19192, 26961, 27738,
 -32766, 15671, 13867, 23124, -32766, -3218, -25583, -12293, 11254, 29016,
 -32766, -17501, 32763, 32763, -32766, 9802, -18306, 2759, -31651, 32765,
 -31432, 32766, -28673, 3121, -32766, 6631, -12155, -782, -32678, 6496,
 -2996, 5945, -32645, 32762, -16977, 1577, 23567, 3730, -1506, -32766,
 22975, -32766, 16383, 2289, -32766, -30169, -8970, -18684, 2505, -168,
 -32766, 11807, 32767, -32391, 11978, 24541, -16385, -32766, -19498, 16383,
 -8709, -32766, 32767, 24735, -16650, -18966, -21828, 25681, 4056, -5465,
 29369, -16288, -31887, -13644, 17530, -31574, 26584, -12568, -276, -13577,
 32766, -32768, 32767, -32359, -32768, -26852, -14794, 32763, -10479,
 -28671, -7066, 3280, -6402, 8795, 3464, -11328, -11789, 15272, 333, 4457,
 -31825, 26867, -2992, 3062, 6399, -3023, 32765, -32766, 4816, -26883,
 -9713, 5277, 5163, -32766, 23270, -9057, 4191, 10918, -30999, 13558,
 32766, 32767, 32760, -32768, -32768, 26606, -31394, 32765, -32768, 32767,
 32765, -32768, -32768, 25597, 32766, -4644, 32753, 18086, -31930, 12193,
 -10920, -4127, 7508, 32765, 32765, -30770, -32060, -32768, 32242, -32768,
 32767, -29866, 30819, -32768, 32765, -28444, 16156, -5568, -13951, 32757,
 -351, -7680, 32764, -32766, -31213, 2619, -4887, 19831, 32762, -28671,
 32758, 32766, 32755, 22027, 9145, 14719, -31166, -29118, 15677, -32132,
 32766, -32768, -32768, -32768, 32765, -32768, -32768, 5488, -28822,
 -31855, 32760, 32762, 32764, 32760, 16383, -3166, 1385, -3225, 5573, 7883,
 14858, 13546, -13510, -7994, 32765, 32767, -32768, -6116 };

static const uint16_t MODEL_NEURONS_LINKS[] = {
 1, 16, 19, 20, 28, 29, 42, 43, 45, 46, 50, 54, 63, 64, 73, 74, 84, 31, 36,
 46, 54, 58, 79, 84, 12, 17, 31, 38, 52, 70, 71, 72, 75, 81, 84, 1, 19, 27,
 36, 43, 49, 67, 71, 72, 78, 81, 84, 8, 15, 21, 23, 26, 29, 35, 37, 39, 41,
 42, 50, 54, 58, 66, 68, 73, 74, 84, 0, 6, 15, 16, 17, 19, 21, 22, 28, 32,
 33, 41, 45, 48, 51, 74, 75, 84, 0, 1, 5, 0, 1, 2, 9, 16, 17, 19, 20, 23,
 25, 30, 31, 32, 33, 34, 35, 40, 48, 54, 57, 61, 66, 67, 70, 72, 80, 81,
 82, 84, 16, 20, 30, 44, 48, 55, 57, 66, 74, 84, 0, 1, 5, 6, 2, 8, 15, 16,
 17, 24, 25, 32, 34, 41, 76, 84, 0, 4, 11, 16, 46, 48, 50, 52, 54, 64, 80,
 84, 2, 6, 0, 10, 15, 24, 29, 61, 70, 72, 78, 84, 3, 10, 21, 38, 53, 66,
 71, 84, 3, 11, 84, 4, 84, 4, 13, 84, 3, 5, 9, 6, 16, 17, 44, 45, 48, 52,
 66, 82, 84, 6, 7, 13, 33, 34, 42, 44, 57, 61, 67, 84, 7, 16, 31, 32, 84,
 6, 15, 2, 5, 12, 14, 17, 18, 27, 31, 41, 44, 76, 83, 84, 3, 6, 18, 17, 27,
 49, 69, 84, 6, 18, 9, 14, 29, 44, 83, 84, 6, 7, 17, 0, 27, 57, 84, 6, 7,
 17, 13, 57, 76, 77, 84, 1, 7, 17, 18, 34, 44, 58, 67, 84, 1, 7, 21, 23, 3,
 8, 14, 24, 30, 36, 52, 65, 84, 2, 11, 36, 68, 84, 1, 7, 18, 23, 29, 63,
 67, 84, 18, 23, 39, 47, 63, 64, 79, 84, 20, 11, 16, 28, 29, 30, 54, 84, 0,
 15, 18, 19, 3, 22, 45, 46, 56, 59, 69, 84, 3, 18, 23, 8, 39, 47, 64, 84,
 0, 9, 15, 18, 29, 20, 22, 36, 50, 84, 18, 20, 24, 25, 11, 26, 40, 44, 51,
 55, 57, 64, 67, 68, 84, 17, 20, 25, 0, 2, 16, 23, 26, 51, 54, 56, 71, 80,
 84, 5, 18, 25, 5, 21, 28, 31, 69, 84, 6, 15, 16, 17, 18, 22, 25, 2, 17,
 23, 55, 57, 62, 84, 6, 15, 18, 2, 7, 24, 62, 84, 15, 18, 1, 34, 84, 6, 7,
 15, 18, 33, 4, 5, 16, 24, 35, 44, 62, 84, 0, 15, 18, 19, 29, 31, 22, 76,
 84, 0, 9, 29, 31, 39, 84, 15, 18, 0, 1, 28, 29, 35, 84, 17, 41, 8, 13, 23,
 80, 84, 15, 19, 31, 34, 39, 2, 8, 13, 15, 23, 48, 60, 65, 84, 5, 8, 15,
 23, 35, 39, 48, 76, 77, 84, 6, 15, 33, 35, 3, 4, 16, 44, 84, 41, 1, 15,
 23, 44, 51, 61, 84, 5, 15, 43, 44, 46, 84, 0, 2, 18, 41, 0, 35, 37, 84,
 18, 24, 38, 41, 5, 18, 35, 37, 62, 84, 18, 22, 41, 49, 3, 37, 39, 84, 18,
 19, 22, 35, 3, 64, 72, 84, 1, 8, 18, 19, 20, 23, 26, 27, 30, 37, 41, 48,
 49, 50, 51, 84, 17, 46, 12, 13, 20, 30, 84, 7, 17, 21, 22, 24, 25, 28, 32,
 33, 34, 42, 53, 84, 6, 15, 18, 35, 1, 4, 16, 35, 44, 84, 0, 6, 18, 20, 38,
 51, 55, 35, 62, 67, 68, 72, 84, 6, 16, 35, 36, 38, 45, 55, 56, 84, 2, 10,
 11, 17, 34, 37, 48, 2, 16, 25, 70, 72, 77, 81, 83, 84, 2, 10, 58, 84 };

static const uint16_t MODEL_NEURON_INTERNAL_LINKS_NUM[] = {
 0, 17, 24, 36, 47, 67, 87, 116, 130, 143, 156, 168, 176, 178, 181, 185,
 197, 208, 213, 229, 236, 245, 252, 261, 270, 279, 288, 294, 301, 312, 323,
 333, 342, 356, 370, 383, 393, 400, 408, 422, 430, 433, 441, 451, 461, 474,
 480, 492, 497, 505, 515, 523, 542, 545, 562, 567, 580, 594, 602, 614 };

static const uint16_t MODEL_NEURON_EXTERNAL_LINKS_NUM[] = {
 17, 24, 35, 47, 66, 84, 116, 126, 142, 154, 166, 174, 177, 179, 182, 195,
 206, 211, 226, 234, 242, 249, 257, 266, 279, 284, 292, 300, 308, 320, 328,
 338, 353, 367, 376, 390, 398, 403, 416, 425, 431, 439, 446, 460, 470, 479,
 487, 493, 501, 511, 519, 527, 543, 550, 563, 573, 586, 595, 611, 615 };

static const nrf_user_coeff_t MODEL_NEURON_ACTIVATION_WEIGHTS[] = {
 0, 0, 0, 0, 0, 0, 0, 0, 0, 63488, 55296, 58368, 40953, 63488, 40960,
 64512, 0, 64512, 0, 0, 0, 64512, 64512, 0, 64512, 64512, 0, 0, 64512,
 63488, 0, 63488, 64512, 64512, 64512, 0, 0, 0, 0, 63488, 40960, 0, 64512,
 64512, 64512, 0, 64512, 40959, 0, 0, 0, 0, 40960, 64512, 40959, 0, 0,
 40960, 55296, 39515 };

static const uint8_t MODEL_NEURON_ACTIVATION_TYPE_MASK[] = {
 0xff, 0xaf, 0xff, 0xff, 0xff, 0x7e, 0xaf, 0x5 };

static const uint16_t MODEL_OUTPUT_NEURONS_INDICES[] = {
 40, 52, 59, 12, 14, 47, 57, 54 };


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
