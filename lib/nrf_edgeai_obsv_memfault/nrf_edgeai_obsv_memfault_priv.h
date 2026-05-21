/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef NRF_EDGEAI_OBSV_MEMFAULT_PRIV_H
#define NRF_EDGEAI_OBSV_MEMFAULT_PRIV_H

/*
 * Staging layout and external linkage for CONFIG_ZTEST. Production code should
 * use nrf_edgeai_obsv_memfault.h only.
 */

#include <stdbool.h>
#include <stdint.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_encode_sizes.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

/** @brief Staging buffer size for the Memfault CDR transport. */
#define NRF_EDGEAI_OBSV_ENCODE_LIST_BUFSZ \
	NRF_EDGEAI_OBSV_ENCODE_LIST_BUF_SIZE(CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS)

struct nrf_edgeai_obsv_ctx;

typedef struct nrf_edgeai_obsv_mflt_staging {
	/** @brief Registered observability contexts (up to MAX_CONTEXTS). */
	struct nrf_edgeai_obsv_ctx *ctxs[CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_MAX_CONTEXTS];
	/** @brief Number of registered contexts. */
	uint8_t num_ctxs;
	/**
	 * @brief Staged CDR payload: obsv-list = [+ obsv-payload].
	 *
	 * Sized for MAX_CONTEXTS payloads plus a small array-header overhead.
	 */
	uint8_t buf[NRF_EDGEAI_OBSV_ENCODE_LIST_BUFSZ];
	uint16_t len;
	bool ready;
	uint32_t staged_duration_ms;
	uint32_t last_collect_ms;
} nrf_edgeai_obsv_mflt_staging_t;

#endif /* NRF_EDGEAI_OBSV_MEMFAULT_PRIV_H */
