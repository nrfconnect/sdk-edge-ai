/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/logging/log.h>
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai/rt/nrf_edgeai_runtime_aux.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#include "../dmic.h"
#include "../model_utils.h"
#include "kws.h"
#include "model_wiring.h"
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

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY)

static nrf_edgeai_obsv_ctx_t kws_ctx;

static uint32_t kws_pd_buf[NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(KEYWORDS_COUNT) / sizeof(uint32_t)];
static uint32_t kws_tm_buf[NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(KEYWORDS_COUNT) / sizeof(uint32_t)];
static nrf_edgeai_obsv_metric_t kws_pd;
static nrf_edgeai_obsv_metric_t kws_tm;

BUILD_ASSERT(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES >= KEYWORDS_COUNT,
	     "Observability will not fit all keyword spotting classes");

static int kws_obsv_init(nrf_edgeai_t *model)
{
	nrf_edgeai_obsv_model_info_t info;
	int err;

	err = obsv_model_info_from_model(model, KEYWORDS_COUNT, &info);
	if (err) {
		return err;
	}

	err = nrf_edgeai_obsv_init(&kws_ctx, &info);
	if (err) {
		LOG_ERR("Observability init failed (err %d)", err);
		return err;
	}

	nrf_edgeai_obsv_metric_pd_create(&kws_pd, kws_pd_buf, KEYWORDS_COUNT);
	nrf_edgeai_obsv_metric_tm_create(&kws_tm, kws_tm_buf, KEYWORDS_COUNT);
	err = nrf_edgeai_obsv_register(&kws_ctx, &kws_pd, NULL);
	if (err) {
		LOG_ERR("PD metric registration failed (err %d)", err);
		return err;
	}

	err = nrf_edgeai_obsv_register(&kws_ctx, &kws_tm, NULL);
	if (err) {
		LOG_ERR("TM metric registration failed (err %d)", err);
		return err;
	}

	err = nrf_edgeai_obsv_memfault_init(&kws_ctx);
	if (err) {
		LOG_ERR("Memfault transport init failed (err %d)", err);
		return err;
	}

	return 0;
}

#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY) */

int kws_init(void)
{
	kws_model = kws_model_ota_load();
	if (kws_model == NULL) {
		LOG_ERR("No usable KWS model - see model_storage_kws flashing instructions in "
			"doc/libraries/model_ota.rst");
		return -ENOENT;
	}
	__ASSERT_NO_MSG(nrf_edgeai_model_outputs_num(kws_model) == KEYWORDS_COUNT);
	__ASSERT_NO_MSG(nrf_edgeai_input_window_size(kws_model) == DMIC_SAMPLES_IN_BLOCK);

	nrf_edgeai_err_t err = nrf_edgeai_init(kws_model);

	if (err) {
		LOG_ERR("Model initialization failed (err %d)", err);
		return -ENOENT;
	}

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY)
	return kws_obsv_init(kws_model);
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY) */

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

	err = nrf_edgeai_run_inference(kws_model);
	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Skip output extraction, not enough data. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to run inference (err %d)", err);
		return -EPERM;
	}

	kws_postprocess(prediction);

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY)
	err = nrf_edgeai_obsv_update_probs(&kws_ctx,
					   kws_model->decoded_output.classif.probabilities.p_f32);
	if (err) {
		LOG_ERR("Failed to update obsv (err %d)", err);
	}
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY) */

	return 0;
}

void kws_reset(void)
{
	nrf_edgeai_model_axon_init_persistent_vars(kws_model);
}
