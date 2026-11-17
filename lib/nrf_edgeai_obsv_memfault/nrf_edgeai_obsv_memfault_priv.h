/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef NRF_EDGEAI_OBSV_MEMFAULT_PRIV_H
#define NRF_EDGEAI_OBSV_MEMFAULT_PRIV_H

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_memfault.h>

struct nrf_edgeai_obsv_mflt_registration {
	struct nrf_edgeai_obsv_ctx *ctx;
	const struct nrf_edgeai_obsv_memfault_keys *keys;
};

#endif /* NRF_EDGEAI_OBSV_MEMFAULT_PRIV_H */
