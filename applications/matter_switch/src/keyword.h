/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef __KEYWORD_H__
#define __KEYWORD_H__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct {
	const char *name;
	size_t count_needed;
	uint8_t threshold_percent;
} keyword_class_cfg_t;

typedef struct {
	bool has_active_class;
	uint16_t predicted_class;
	size_t count;
	size_t non_keyword_count;
	uint16_t non_keyword_class;
	float average_probability;
	bool wait_for_class_change;
	uint16_t blocked_class;
} keyword_runtime_ctx_t;

typedef struct {
	bool has_first_keyword;
	uint16_t first_class;
	float first_probability;
	uint32_t detected_at_ms;
} keyword_phrase_ctx_t;

typedef enum keyword_labels_e {
	KEYWORD_OFF = 0,
	KEYWORD_ON = 1,
	KEYWORD_OTHER = 2,
	KEYWORD_SILENCE = 3,
	KEYWORD_SWITCH = 4,
	KEYWORDS_cnt
} keyword_labels_t;

int kw_init(void);
void kw_reset_model(void);
/**
 * @return 0 while still listening, 1 when a full keyword command is recognized (see @a kw_class),
 *         -EBUSY if more audio is needed, or another negative errno on error.
 */
int kw_process(uint8_t *const audio_buffer, const uint16_t num_samples, uint16_t *const kw_class);

extern const keyword_class_cfg_t KEYWORD_CLASSES_CFG[];
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif