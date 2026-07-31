/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef MEMFAULT_FOTA_H_
#define MEMFAULT_FOTA_H_

#include <zephyr/sys/util.h>

/**
 * @{
 * @ingroup ww_kws
 */

#if IS_ENABLED(CONFIG_WW_KWS_MEMFAULT_FOTA)

/**
 * @brief Populate Bluetooth DIS serial from HW ID for Memfault FOTA.
 *
 * Must be called after @c settings_load().
 *
 * @return 0 on success, -errno on failure.
 */
int memfault_fota_init(void);

#else

static inline int memfault_fota_init(void)
{
	return 0;
}

#endif /* IS_ENABLED(CONFIG_WW_KWS_MEMFAULT_FOTA) */

/**
 * @}
 */

#endif /* MEMFAULT_FOTA_H_ */
