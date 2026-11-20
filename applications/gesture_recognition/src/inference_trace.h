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
 * @brief GPIO trace line for power measurements.
 *
 * The line (@c infer-trace-gpios in the @c zephyr,user node of the board
 * overlay) is driven high while the model runs inference, so a logic analyzer
 * or the Power Profiler Kit II digital inputs can gate the current measurement
 * on it.
 */

#if IS_ENABLED(CONFIG_INFERENCE_GPIO_TRACE)

/**
 * @brief Drive the trace line.
 *
 * @param active True to drive the line high, false to drive it low.
 */
void inference_trace_set(bool active);

#else

static inline void inference_trace_set(bool active)
{
}

#endif /* IS_ENABLED(CONFIG_INFERENCE_GPIO_TRACE) */

#endif /* INFERENCE_TRACE_H */
