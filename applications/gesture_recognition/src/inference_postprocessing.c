// ///////////////////////// Package Header Files ////////////////////////////
#include "inference_postprocessing.h"

// ////////////////////// Standard C++ Header Files //////////////////////////
// /////////////////////// Standard C Header Files ///////////////////////////
#include <string.h>

///
#define PREVIOUS_PREDICTION_NUM                 (3)
///

//////////////////////////////////////////////////////////////////////////////

typedef struct prediction_ctx_s
{
    /** Prediction target */
    uint16_t target;

    /** Prediction probability */
    float probability;
} prediction_ctx_t;

typedef struct prediction_tracer_s
{
    /** Current prediction index */
    int index;

    /** Current prediction target */
    uint16_t target;

    /** Previous predictions context */
    prediction_ctx_t prev[PREVIOUS_PREDICTION_NUM];
} prediction_tracer_t;

/**
 * @brief Class prediction conditions for postprocessing
 * 
 */
typedef struct class_prediction_condition_s
{
    /** Minimum number of repetitions of a class for prediction */
    uint16_t min_repeat_count;

    /** Minimum probability threshold for prediction */
    float probability_threshold;
} class_prediction_condition_t;

//////////////////////////////////////////////////////////////////////////////

static const class_prediction_condition_t* get_class_condition_(uint8_t predicted_target);
static const char* get_name_by_target_(uint8_t predicted_target);

//////////////////////////////////////////////////////////////////////////////

void inference_postprocess(const uint16_t predicted_target,
                            const float prob,
                            const bool do_postprocessing,
                            inference_postprocess_cb_t callback)
{
    uint16_t target = predicted_target;
    float probability = prob;
    
    static prediction_tracer_t tracer_ = {0U};

    if (do_postprocessing) {

        if ((target == CLASS_LABEL_UNKNOWN) || (target == CLASS_LABEL_IDLE)) {
            /** Reset tracer for UNKNOWN and IDLE classes */
            tracer_.index = 0;
            tracer_.target = target;
        } else {
            if (tracer_.index >= PREVIOUS_PREDICTION_NUM)
                tracer_.index = 0;

            /** Reset tracer if predicted class is not the same as previous */
            if (tracer_.target != target) {
                tracer_.index = 0;
                tracer_.target = target;
            }

            tracer_.prev[tracer_.index].probability = probability;
            tracer_.index++;

            const class_prediction_condition_t* class_condition = get_class_condition_(target);
            if (class_condition == NULL) {
                return;
            }

            /** Сlass is labled as CLASS_LABEL_UNKNOWN if the number of repetitions does not exceed the threshold */
            if (tracer_.index >= class_condition->min_repeat_count) {
                /** Calculate average probability for last N predictions of the same class */
                float average_prob = 0.0f;

                for (int i = 0; i < tracer_.index; ++i)
                    average_prob += tracer_.prev[i].probability;

                average_prob = average_prob / tracer_.index;

                /** If average probability is less the class probability threshold,
                 * the class is labled as CLASS_LABEL_UNKNOWN */
                if (average_prob < class_condition->probability_threshold)
                    target = CLASS_LABEL_UNKNOWN;
                else
                    probability = average_prob;

                /** Reset tracer index for non-repetative classes */
                if ((target != CLASS_LABEL_ROTATION_RIGHT) || (target != CLASS_LABEL_ROTATION_LEFT))
                    tracer_.index = 0;
            } else {
                target = CLASS_LABEL_UNKNOWN;
            }
        }
    }

    /** Provide result to user callback */
    if (callback)
        callback(target, probability, get_name_by_target_(target), !do_postprocessing);
}

//////////////////////////////////////////////////////////////////////////////

static const char* get_name_by_target_(uint8_t predicted_target)
{

    static const char* LABEL_VS_NAME[] = 
    {
        [CLASS_LABEL_IDLE]           = "IDLE",
        [CLASS_LABEL_UNKNOWN]        = "UNKNOWN",
        [CLASS_LABEL_SWIPE_LEFT]     = "SWIPE LEFT",
        [CLASS_LABEL_SWIPE_RIGHT]    = "SWIPE RIGHT",
        [CLASS_LABEL_DOUBLE_SHAKE]   = "DOUBLE SHAKE",
        [CLASS_LABEL_DOUBLE_THUMB]   = "DOUBLE THUMB",
        [CLASS_LABEL_ROTATION_RIGHT] = "ROTATION RIGHT",
        [CLASS_LABEL_ROTATION_LEFT]  = "ROTATION LEFT"
    };

    static const uint8_t LABELS_CNT = sizeof(LABEL_VS_NAME) / sizeof(LABEL_VS_NAME[0]);

    return (predicted_target < LABELS_CNT) ? LABEL_VS_NAME[predicted_target] : NULL;
}

//////////////////////////////////////////////////////////////////////////////

static const class_prediction_condition_t* get_class_condition_(uint8_t predicted_target)
{
    static const class_prediction_condition_t LABEL_VS_CONFIG[] = 
    {
        [CLASS_LABEL_IDLE]           = {0, 0.0},
        [CLASS_LABEL_UNKNOWN]        = {0, 0.0},
        [CLASS_LABEL_SWIPE_LEFT]     = {2, 0.8},
        [CLASS_LABEL_SWIPE_RIGHT]    = {2, 0.8},
        [CLASS_LABEL_DOUBLE_SHAKE]   = {2, 0.7},
        [CLASS_LABEL_DOUBLE_THUMB]   = {2, 0.7},
        [CLASS_LABEL_ROTATION_RIGHT] = {2, 0.7},
        [CLASS_LABEL_ROTATION_LEFT]  = {2, 0.7},
    };

    static const uint8_t LABELS_CNT = sizeof(LABEL_VS_CONFIG) / sizeof(LABEL_VS_CONFIG[0]);

    return (predicted_target < LABELS_CNT) ? &LABEL_VS_CONFIG[predicted_target] : NULL;
}
