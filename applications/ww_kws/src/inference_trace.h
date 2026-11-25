/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef INFERENCE_TRACE_H
#define INFERENCE_TRACE_H

#include <stdbool.h>
#include <zephyr/sys/util_macro.h>

/**
 * @brief GPIO trace lines for power measurements.
 *
 * Each line is driven high while the matching model runs inference, so a
 * logic analyzer or the Power Profiler Kit II digital inputs can gate the
 * current measurement on it. The pins come from the @c zephyr,user node in the
 * board overlay.
 */
enum inference_trace_line {
	/** Wake word model (@c ww-trace-gpios). */
	INFERENCE_TRACE_WW,
	/** Keyword spotting model (@c kws-trace-gpios). */
	INFERENCE_TRACE_KWS,
};

#if IS_ENABLED(CONFIG_INFERENCE_GPIO_TRACE)

/**
 * @brief Drive a trace line.
 *
 * @param line Trace line to drive.
 * @param active True to drive the line high, false to drive it low.
 */
void inference_trace_set(enum inference_trace_line line, bool active);

#else

static inline void inference_trace_set(enum inference_trace_line line, bool active)
{
}

#endif /* IS_ENABLED(CONFIG_INFERENCE_GPIO_TRACE) */

#endif /* INFERENCE_TRACE_H */
