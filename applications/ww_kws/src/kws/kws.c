/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/logging/log.h>
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai/rt/nrf_edgeai_runtime.h>
#include <nrf_edgeai/rt/nrf_edgeai_runtime_aux.h>

#include "../dmic.h"
#include "../model_utils.h"
#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS)
#include "../obsv/model_obsv.h"
#endif
#include "kws.h"
#include "nrf_edgeai_generated/nrf_edgeai_user_model.h"
#include "nrf_edgeai_generated/nrf_edgeai_user_model_labels.h"

LOG_MODULE_REGISTER(kws);

/* Equal to 300 ms of audio. */
#define SKIP_DETECTIONS_COUNT 10

#define KEYWORDS_COUNT ARRAY_SIZE(keyword_detection_ctxs)

struct keyword_detection_ctx {
	const float threshold;
	const uint8_t num_in_row;
};

static const struct keyword_detection_ctx keyword_detection_ctxs[] = {
	[MODEL_LABEL_INDEX_OTHER] = {},
	[MODEL_LABEL_INDEX_SILENCE] = {},
	[MODEL_LABEL_INDEX_DOWN] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_GO] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_LEFT] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_NO] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_OFF] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_ON] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_RIGHT] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_STOP] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_UP] = {.threshold = 0.8f, .num_in_row = 10},
	[MODEL_LABEL_INDEX_YES] = {.threshold = 0.8f, .num_in_row = 10},
};

static nrf_edgeai_t *kws_model;

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS)

/* Mel feature vector length produced by the model DSP front end. Sizes the
 * FEATURES-stream metric storage; validated at runtime against
 * nrf_edgeai_dsp_features_ctx()->overall_num.
 */
#define KWS_NUM_FEATURES 40

static struct model_obsv kws_obsv;

BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES >= KEYWORDS_COUNT,
	     "Observability will not fit all keyword spotting classes");
BUILD_ASSERT(KWS_NUM_FEATURES <= MODEL_OBSV_MAX_FEATURES,
	     "MODEL_OBSV_MAX_FEATURES must be >= KWS_NUM_FEATURES");

static int kws_obsv_init(nrf_edgeai_t *model)
{
	nrf_edgeai_obsv_model_info_t info;
	int err;

	err = obsv_model_info_from_model(model, KEYWORDS_COUNT, &info);
	if (err) {
		return err;
	}

	return model_obsv_init(&kws_obsv, &info, KWS_NUM_FEATURES);
}

#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS) */

int kws_init(void)
{
	kws_model = nrf_edgeai_user_model_36712();
	__ASSERT_NO_MSG(kws_model);
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(kws_model) == KEYWORDS_COUNT);
	__ASSERT_NO_MSG(nrf_edgeai_input_window_size(kws_model) == DMIC_SAMPLES_IN_BLOCK);

	nrf_edgeai_err_t err = nrf_edgeai_init(kws_model);

	if (err) {
		LOG_ERR("Model initialization failed (err %d)", err);
		return -ENOENT;
	}

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS)
	return kws_obsv_init(kws_model);
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS) */

	return 0;
}

static void kws_postprocess(struct kws_prediction *const prediction)
{
	prediction->valid = false;

	const float alpha = CONFIG_KWS_EMA_ALPHA / 1000.0f;
	static enum nrf_edgeai_user_label_e last_class;
	static int count;

	/* Exponential moving average of class probability. */
	static float probability_ema;

	const uint16_t predicted_class = kws_model->decoded_output.classif.predicted_class;

	__ASSERT_NO_MSG(predicted_class < KEYWORDS_COUNT);

	const flt32_t class_probability =
		kws_model->decoded_output.classif.probabilities.p_f32[predicted_class];
	const struct keyword_detection_ctx *class_ctx = &keyword_detection_ctxs[predicted_class];
	const char *class_name = NRF_EDGEAI_USER_LABELS_NAME[predicted_class];

	if (predicted_class == MODEL_LABEL_INDEX_OTHER ||
	    predicted_class == MODEL_LABEL_INDEX_SILENCE) {
		LOG_DBG("class: %s, prob: %f", class_name, (double)class_probability);

		count = 0;
		probability_ema = 0.0f;
		return;
	}

	if (predicted_class != last_class) {
		last_class = predicted_class;
		count = 0;
		probability_ema = 0.0f;
	}

	count++;
	probability_ema = alpha * class_probability + (1 - alpha) * probability_ema;

	LOG_DBG("class: %s, count %d, prob: %f, ema %f", class_name, count,
		(double)class_probability, (double)probability_ema);

	if (count >= class_ctx->num_in_row && probability_ema >= class_ctx->threshold) {
		prediction->valid = true;
		prediction->class = predicted_class;
		prediction->avg_probability = probability_ema;
		prediction->name = class_name;

		/* Skip detections to reduce double spotting. */
		count = -SKIP_DETECTIONS_COUNT;
		probability_ema = 0.0f;
	}
}

int kws_process(uint8_t *const audio_buffer, const uint16_t num_samples,
		struct kws_prediction *const prediction)
{
	__ASSERT_NO_MSG(audio_buffer);
	__ASSERT_NO_MSG(num_samples == nrf_edgeai_input_window_size(kws_model));
	__ASSERT_NO_MSG(prediction);

	nrf_edgeai_err_t err;

	err = nrf_edgeai_feed_inputs(kws_model, audio_buffer, num_samples);
	free_dmic_buffer(audio_buffer);

	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Skip inference, not enough data. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to feed inputs (err %d)", err);
		return -EPERM;
	}

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS)
	/* Extract the mel feature vector before inference and feed it to the
	 * FEATURES-stream metrics (mel energy / spectral descriptors). run_inference
	 * reuses these features, so the explicit call adds no extra DSP work.
	 */
	err = nrf_edgeai_process_features(kws_model);
	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Feature window not complete yet. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to process features (err %d)", err);
		return -EPERM;
	}

	const nrf_edgeai_dsp_feature_extraction_t *feats = nrf_edgeai_dsp_features_ctx(kws_model);

	if (feats != NULL) {
		model_obsv_update_features(&kws_obsv, feats->buffer.p_f32, feats->overall_num);
	}
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS) */

	err = nrf_edgeai_run_inference(kws_model);
	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Skip output extraction, not enough data. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to run inference (err %d)", err);
		return -EPERM;
	}

	kws_postprocess(prediction);

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS)
	model_obsv_update_probs(&kws_obsv, kws_model->decoded_output.classif.probabilities.p_f32);
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_KWS) */

	return 0;
}

void kws_reset(void)
{
	nrf_edgeai_model_axon_init_persistent_vars(kws_model);
}
