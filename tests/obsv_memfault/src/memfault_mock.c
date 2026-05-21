/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/*
 * Minimal stub for Memfault CDR registration.
 *
 * nrf_edgeai_obsv_memfault_init() calls memfault_cdr_register_source() once to
 * register the CDR source with the Memfault SDK. In this test build,
 * CONFIG_MEMFAULT_CDR_ENABLE=n, so the real implementation is not compiled.
 * This stub satisfies the linker and allows testing all CDR source callbacks
 * (has_cdr_cb, read_data_cb, mark_cdr_read_cb) directly without a live
 * Memfault transport.
 */

#include <memfault/core/custom_data_recording.h>

bool memfault_cdr_register_source(const sMemfaultCdrSourceImpl *source)
{
	return true;
}
