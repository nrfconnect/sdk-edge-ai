/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef BLE_APP_H_
#define BLE_APP_H_

#include <zephyr/sys/util.h>

/**
 * @{
 * @ingroup ww_kws
 */

#if IS_ENABLED(CONFIG_BT)

/**
 * @brief Initialize Bluetooth and start connectable advertising.
 *
 * Advertises the SMP service when CONFIG_MCUMGR_TRANSPORT_BT is enabled and/or
 * the Memfault Diagnostic Service when CONFIG_MODELS_OBSERVABILITY_MDS is enabled.
 *
 * @return 0 on success, -errno on failure.
 */
int init_app_ble(void);

#else

static inline int init_app_ble(void)
{
	return 0;
}

#endif /* IS_ENABLED(CONFIG_BT) */

/**
 * @}
 */

#endif /* BLE_APP_H_ */
