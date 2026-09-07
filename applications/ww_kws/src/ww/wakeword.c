/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/logging/log.h>
#include <zephyr/sys/__assert.h>
#include <zephyr/sys/util.h>
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai/rt/nrf_edgeai_runtime.h>
#include <nrf_edgeai/rt/nrf_edgeai_runtime_aux.h>

#include "../dmic.h"
#include "nrf_edgeai_generated/nrf_edgeai_user_model.h"
#include "wakeword.h"

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_WW)
#include "ww_obsv.h"
#endif

LOG_MODULE_REGISTER(ww);

static nrf_edgeai_t *ww_model;

int ww_init(void)
{
	ww_model = nrf_edgeai_user_model_36711();
	__ASSERT_NO_MSG(ww_model);
	__ASSERT_NO_MSG(ww_model->input.window_size == DMIC_SAMPLES_IN_BLOCK);

	nrf_edgeai_err_t err = nrf_edgeai_init(ww_model);

	if (err) {
		LOG_ERR("Model initialization failed (err %d)", err);
		return -ENOENT;
	}

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_WW)
	return ww_obsv_init(ww_model);
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_WW) */

	return 0;
}

static bool ww_postprocess(void)
{
	static uint32_t ww_count;
	static uint32_t ww_history;

	const float ww_threshold = CONFIG_WW_PROBABILITY_THRESHOLD / 1000.f;

	const uint16_t predicted_class = ww_model->decoded_output.classif.predicted_class;
	const float class_probability =
		ww_model->decoded_output.classif.probabilities.p_f32[predicted_class];
	const bool ww_detected = class_probability > ww_threshold;

	const bool oldest_entry = (bool)(ww_history & BIT(CONFIG_WW_HISTORY_SIZE - 1));

	ww_count = ww_count + ww_detected - oldest_entry;
	ww_history = (ww_history << 1) | ww_detected;

	LOG_DBG("postprocess: count: %2u, probability: %f", ww_count, (double)class_probability);

	if (ww_count >= CONFIG_WW_COUNT_THRESHOLD) {
		ww_count = 0;
		ww_history = 0;

		return true;
	}

	return false;
}

int ww_process(uint8_t *const audio_buffer, const uint16_t num_samples, bool *const ww_detected)
{
	__ASSERT_NO_MSG(audio_buffer);
	__ASSERT_NO_MSG(num_samples == nrf_edgeai_input_window_size(ww_model));
	__ASSERT_NO_MSG(ww_detected);

	nrf_edgeai_err_t err;

	err = nrf_edgeai_feed_inputs(ww_model, audio_buffer, num_samples);
	free_dmic_buffer(audio_buffer);

	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Skip inference, not enough data. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to feed inputs (err %d)", err);
		return -EPERM;
	}

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_WW)
	/* Extract the mel feature vector before inference for the FEATURES-stream
	 * metrics. run_inference reuses these features, so the explicit call adds no
	 * extra DSP work.
	 */
	err = nrf_edgeai_process_features(ww_model);
	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Feature window not complete yet. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to process features (err %d)", err);
		return -EPERM;
	}

	const nrf_edgeai_dsp_feature_extraction_t *feats = nrf_edgeai_dsp_features_ctx(ww_model);

	if (feats != NULL) {
		ww_obsv_update_features(feats->buffer.p_f32, feats->overall_num);
	}
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_WW) */

	err = nrf_edgeai_run_inference(ww_model);
	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Skip output extraction, not enough data. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to run inference (err %d)", err);
		return -EPERM;
	}

	*ww_detected = ww_postprocess();

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_WW)
	ww_obsv_update_probs(ww_model->decoded_output.classif.probabilities.p_f32[0]);
#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_WW) */

	return 0;
}

void ww_reset(void)
{
	nrf_edgeai_model_axon_init_persistent_vars(ww_model);
}
