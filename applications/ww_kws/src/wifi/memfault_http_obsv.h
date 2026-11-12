/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MEMFAULT_HTTP_OBSV_H_
#define MEMFAULT_HTTP_OBSV_H_

/**
 * @{
 * @ingroup ww_kws
 */

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_MEMFAULT_HTTP)

/**
 * @brief Start the network connection used to upload observability data.
 *
 * Brings up the network interfaces and enables Memfault periodic upload over HTTPS
 * while the network is connected.
 *
 * @return 0 on success, -errno on failure.
 */
int init_app_memfault_http(void);

#else

static inline int init_app_memfault_http(void)
{
	return 0;
}

#endif /* CONFIG_MODELS_OBSERVABILITY_MEMFAULT_HTTP */

/**
 * @}
 */

#endif /* MEMFAULT_HTTP_OBSV_H_ */
