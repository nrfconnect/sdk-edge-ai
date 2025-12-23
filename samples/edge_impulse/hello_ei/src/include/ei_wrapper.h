/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/** @file
 * @brief Edge Impulse wrapper header.
 */

#ifndef _EI_WRAPPER_H_
#define _EI_WRAPPER_H_


/**
 * @defgroup ei_wrapper Edge Impulse wrapper
 * @brief Wrapper that uses Edge Impulse lib to run machine learning on device.
 *
 * @{
 */

#include <zephyr/kernel.h>

#include "ei_classifier_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*get_data_callback_t)(size_t, size_t, float *);

int ei_wrapper_init(get_data_callback_t cbk);

int ei_wrapper_run_inference(ei_impulse_result_t *ei_result, size_t window_size);

#ifdef __cplusplus
}
#endif

/**
 * @}
 */

#endif /* _EI_WRAPPER_H_ */
