/*
* Copyright (c) 2021 Nordic Semiconductor ASA
* SPDX-License-Identifier: Apache-2.0
*/
#ifndef INFERENCE_POSTPROCESSING_H__
#define INFERENCE_POSTPROCESSING_H__

#include <stdint.h>
#include <stdbool.h>

typedef enum
{
    CLASS_LABEL_IDLE,            ///< CLASS_LABEL_IDLE
    CLASS_LABEL_UNKNOWN,         ///< CLASS_LABEL_UNKNOWN
    CLASS_LABEL_SWIPE_RIGHT,     ///< CLASS_LABEL_SWIPE_RIGHT
    CLASS_LABEL_SWIPE_LEFT,      ///< CLASS_LABEL_SWIPE_LEFT
    CLASS_LABEL_DOUBLE_SHAKE,    ///< CLASS_LABEL_DOUBLE_SHAKE
    CLASS_LABEL_DOUBLE_THUMB,    ///< CLASS_LABEL_DOUBLE_THUMB
    CLASS_LABEL_ROTATION_RIGHT,  ///< CLASS_LABEL_ROTATION_RIGHT
    CLASS_LABEL_ROTATION_LEFT,   /// < CLASS_LABEL_ROTATION_LEFT
} class_label_t;

/**
 * @brief Inference Result (prediction) postprocessing callback
 * 
 * @param[in] class_label   Label of the predicted class @ref class_label_t
 * @param[in] probability   Probability of the predicted class   
 * @param[in] class_name    Name of predicted class, null-terminated string
 * @param[in] is_raw        If true the postprocessing logic was not applied to this prediction
 * 
 */
typedef void (*inference_postprocess_cb_t)(const class_label_t class_label, 
                                            const float probability,
                                            const char* class_name,
                                            const bool is_raw);

/**
 * @brief Postprocess the Neuton library RAW inference output
 * 
 * @param[in] predicted_target  Predicted target(class)
 * @param[in] probability       Predicted probability of the target
 * @param[in] do_postprocessing If false, no postprocessing is applied and the raw prediction goes to the user callback unchanged
 * @param[in] callback          Inference Result (prediction) ready user callback, @ref inference_postprocess_cb_t 
 */
void inference_postprocess(const uint16_t predicted_target,
                            const float probability,
                            const bool do_postprocessing,
                            inference_postprocess_cb_t callback);



#endif /* INFERENCE_POSTPROCESSING_H__ */
