/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "keyword.h"

#include <string.h>

#include <zephyr/logging/log.h>
#include <nrf_edgeai/nrf_edgeai.h>
#include <nrf_edgeai/rt/nrf_edgeai_runtime_aux.h>

#include "dmic.h"
#include "models/kws/nrf_edgeai_generated/nrf_edgeai_user_model.h"

LOG_MODULE_DECLARE(kw, CONFIG_CHIP_APP_LOG_LEVEL);

const keyword_class_cfg_t KEYWORD_CLASSES_CFG[] = {
	[KEYWORD_OFF] = {.name = "OFF", .count_needed = 2, .threshold_percent = 0},
	[KEYWORD_ON] = {.name = "ON", .count_needed = 2, .threshold_percent = 0},
	[KEYWORD_OTHER] = {.name = "OTHER", .count_needed = 4, .threshold_percent = 0},
	[KEYWORD_SILENCE] = {.name = "SILENCE", .count_needed = 4, .threshold_percent = 0},
	[KEYWORD_SWITCH] = {.name = "SWITCH", .count_needed = 2, .threshold_percent = 0},
};

static nrf_edgeai_t *kw_model;

static keyword_runtime_ctx_t s_keyword_runtime_ctx;
static keyword_phrase_ctx_t s_keyword_phrase_ctx;

static void reset_keyword_detection_state(void);
static bool is_keyword_command_component(uint16_t predicted_class);
static bool is_keyword_phrase_first_word(uint16_t predicted_class);
static const keyword_class_cfg_t *get_keyword_class_cfg(uint16_t predicted_class);
static bool is_keyword_probability_above_threshold(uint16_t predicted_class, float probability);
static bool try_detect_keyword_command(uint16_t predicted_class, float probability,
				       const char **pp_first_keyword_name,
				       const char **pp_second_keyword_name,
				       bool *p_has_second_keyword, uint8_t *p_average_probability);

int kw_init(void)
{
	kw_model = nrf_edgeai_user_model_kws();
	__ASSERT_NO_MSG(kw_model);

	nrf_edgeai_err_t err = nrf_edgeai_init(kw_model);

	if (err) {
		LOG_ERR("Model initialization failed (err %d)", err);
		return -ENOENT;
	}

	return 0;
}

int kw_process(uint8_t *const audio_buffer, const uint16_t num_samples, uint16_t *const kw_class)
{
	__ASSERT_NO_MSG(kw_model);
	__ASSERT_NO_MSG(audio_buffer);
	__ASSERT_NO_MSG(kw_class);

	nrf_edgeai_err_t err;
	err = nrf_edgeai_feed_inputs(kw_model, audio_buffer, num_samples);

	free_dmic_buffer(audio_buffer);

	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Skip inference, not enough data. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to feed inputs (err %d)", err);
		return -EPERM;
	}

	err = nrf_edgeai_run_inference(kw_model);

	if (err == NRF_EDGEAI_ERR_INPROGRESS) {
		/* Skip output extraction, not enough data. */
		return -EBUSY;
	} else if (err) {
		LOG_ERR("Failed to run inference (err %d)", err);
		return -EPERM;
	}
	// return 1;
	/* KWS model may have more output classes (e.g. 12) than this app enum (KEYWORDS_cnt). */
	const uint16_t raw_class = kw_model->decoded_output.classif.predicted_class;
	const uint16_t n_out = nrf_edgeai_model_outputs_num(kw_model);

	if (n_out == 0U || raw_class >= n_out) {
		*kw_class = KEYWORD_OTHER;
		return 0;
	}
	flt32_t probability = kw_model->decoded_output.classif.probabilities.p_f32[raw_class];
	*kw_class = (raw_class < (uint16_t)KEYWORDS_cnt) ? raw_class : KEYWORD_OTHER;

	const char *first_keyword_name = NULL;
	const char *second_keyword_name = NULL;
	bool has_second_keyword = false;
	uint8_t average_probability_pct = 0;

	if (try_detect_keyword_command(*kw_class, probability, &first_keyword_name,
				       &second_keyword_name, &has_second_keyword,
				       &average_probability_pct)) {
		/* *kw_class still holds the class that completed the command (caller maps to
		 * action). */
		LOG_INF("Keyword detected: %s, %d %%\n", first_keyword_name,
			average_probability_pct);
		reset_keyword_detection_state();
		return 1;
	}
	*kw_class = KEYWORD_OTHER;
	return 0;
}

void kw_reset_model(void)
{
	nrf_edgeai_model_axon_init_persistent_vars(kw_model);
	reset_keyword_detection_state();
}

static void reset_keyword_detection_state(void)
{
	memset(&s_keyword_runtime_ctx, 0, sizeof(s_keyword_runtime_ctx));
	memset(&s_keyword_phrase_ctx, 0, sizeof(s_keyword_phrase_ctx));
}

static bool is_keyword_command_component(uint16_t predicted_class)
{
	return (predicted_class < KEYWORDS_cnt) && (predicted_class != KEYWORD_OTHER) &&
	       (predicted_class != KEYWORD_SILENCE);
}

static bool is_keyword_phrase_first_word(uint16_t predicted_class)
{
	return (predicted_class == KEYWORD_ON) || (predicted_class == KEYWORD_OFF) ||
	       (predicted_class == KEYWORD_SWITCH);
}

static const keyword_class_cfg_t *get_keyword_class_cfg(uint16_t predicted_class)
{
	if (predicted_class >= KEYWORDS_cnt) {
		return &KEYWORD_CLASSES_CFG[KEYWORD_OTHER];
	}
	return &KEYWORD_CLASSES_CFG[predicted_class];
}

static bool is_keyword_probability_above_threshold(uint16_t predicted_class, flt32_t probability)
{
	const keyword_class_cfg_t *p_class_cfg = get_keyword_class_cfg(predicted_class);

	if (p_class_cfg->threshold_percent == 0) {
		return true;
	}

	return (probability * 100.0f) >= (flt32_t)p_class_cfg->threshold_percent;
}

static bool try_detect_keyword_command(uint16_t predicted_class, flt32_t probability,
				       const char **pp_first_keyword_name,
				       const char **pp_second_keyword_name,
				       bool *p_has_second_keyword, uint8_t *p_average_probability)
{
	const keyword_class_cfg_t *p_class_cfg = get_keyword_class_cfg(predicted_class);
	const bool is_command_component = is_keyword_command_component(predicted_class);
	const bool passes_threshold =
		is_command_component &&
		is_keyword_probability_above_threshold(predicted_class, probability);

	*pp_first_keyword_name = NULL;
	*pp_second_keyword_name = NULL;
	*p_has_second_keyword = false;
	*p_average_probability = 0;

	const uint32_t now_ms = k_uptime_get_32();

	if (s_keyword_runtime_ctx.wait_for_class_change) {
		if (predicted_class == s_keyword_runtime_ctx.blocked_class) {
			return false;
		}

		s_keyword_runtime_ctx.wait_for_class_change = false;
	}

	if (!passes_threshold) {
		if (is_command_component) {
			/* A low-confidence keyword should break the current confirmation streak. */
			s_keyword_runtime_ctx.has_active_class = false;
			s_keyword_runtime_ctx.count = 0;
			s_keyword_runtime_ctx.average_probability = 0;
		}

		if ((s_keyword_runtime_ctx.non_keyword_count == 0) ||
		    (s_keyword_runtime_ctx.non_keyword_class != predicted_class)) {
			s_keyword_runtime_ctx.non_keyword_class = predicted_class;
			s_keyword_runtime_ctx.non_keyword_count = 0;
		}

		s_keyword_runtime_ctx.non_keyword_count++;

		if (s_keyword_phrase_ctx.has_first_keyword &&
		    (s_keyword_runtime_ctx.non_keyword_count < p_class_cfg->count_needed)) {
			/* Keep the first word alive across short OTHER/SILENCE gaps. */
			s_keyword_phrase_ctx.detected_at_ms = now_ms;
		}

		if (s_keyword_runtime_ctx.non_keyword_count >= p_class_cfg->count_needed) {
			s_keyword_runtime_ctx.has_active_class = false;
			s_keyword_runtime_ctx.count = 0;
			s_keyword_runtime_ctx.non_keyword_count = 0;
			s_keyword_runtime_ctx.non_keyword_class = 0;
			s_keyword_runtime_ctx.average_probability = 0;
		}

		return false;
	}

	s_keyword_runtime_ctx.non_keyword_count = 0;

	if (!s_keyword_runtime_ctx.has_active_class ||
	    (predicted_class != s_keyword_runtime_ctx.predicted_class)) {
		s_keyword_runtime_ctx.has_active_class = true;
		s_keyword_runtime_ctx.predicted_class = predicted_class;
		s_keyword_runtime_ctx.count = 0;
		s_keyword_runtime_ctx.non_keyword_count = 0;
		s_keyword_runtime_ctx.non_keyword_class = 0;
		s_keyword_runtime_ctx.average_probability = 0;
	}

	s_keyword_runtime_ctx.count++;
	s_keyword_runtime_ctx.average_probability +=
		(probability - s_keyword_runtime_ctx.average_probability) /
		s_keyword_runtime_ctx.count;

	if (s_keyword_phrase_ctx.has_first_keyword) {
		/* Any ongoing keyword activity extends the pairing window. */
		s_keyword_phrase_ctx.detected_at_ms = now_ms;
	}

#if !DEMO_FINAL_DETECTIONS_ONLY && PRINT_KEYWORD_CLASS_ACTIVITY
	printk("Keyword model class %s, count: %u \tprob : %0.3f\n",
	       get_keyword_class_cfg(s_keyword_runtime_ctx.predicted_class)->name,
	       (unsigned int)s_keyword_runtime_ctx.count,
	       s_keyword_runtime_ctx.average_probability);
#endif

	if (s_keyword_runtime_ctx.count < p_class_cfg->count_needed) {
		return false;
	}

	const flt32_t detected_probability = s_keyword_runtime_ctx.average_probability;

	s_keyword_runtime_ctx.has_active_class = false;
	s_keyword_runtime_ctx.count = 0;
	s_keyword_runtime_ctx.non_keyword_count = 0;
	s_keyword_runtime_ctx.non_keyword_class = 0;
	s_keyword_runtime_ctx.average_probability = 0;
	s_keyword_runtime_ctx.wait_for_class_change = true;
	s_keyword_runtime_ctx.blocked_class = predicted_class;

	/* `count_needed` is consecutive stable frames of this class, not a second spoken word. */
	if (is_keyword_phrase_first_word(predicted_class)) {
		unsigned avg_pct = (unsigned)(detected_probability * 100.0f + 0.5f);

		if (avg_pct > 100U) {
			avg_pct = 100U;
		}
		*pp_first_keyword_name = p_class_cfg->name;
		*pp_second_keyword_name = NULL;
		*p_has_second_keyword = false;
		*p_average_probability = (uint8_t)avg_pct;
		return true;
	}

	/* Unsupported combinations are ignored, but the confirmed first word stays active. */
	return false;
}
