/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ei_run_classifier.h"

#if !CONFIG_ZTEST
/* Fixes warnings about redefinition of Zephyr ROUND_UP macro. */
#ifdef ROUND_UP
#undef ROUND_UP
#endif
#endif /* !CONFIG_ZTEST */

#include "ei_wrapper.h"

#define DEBUG_MODE		IS_ENABLED(CONFIG_EI_WRAPPER_DEBUG_MODE)

get_data_callback_t get_data_cbk = nullptr;

int ei_wrapper_init(get_data_callback_t cbk)
{
	get_data_cbk = cbk;
	return 0;
}

int ei_wrapper_run_inference(ei_impulse_result_t *ei_result, size_t window_size)
{
	signal_t features_signal;

	__ASSERT(ei_result != nullptr, "ei_result pointer is null");

	features_signal.get_data = get_data_cbk;
	features_signal.total_length = window_size;

	EI_IMPULSE_ERROR err = run_classifier(&features_signal, ei_result, DEBUG_MODE);

	return err;
}
