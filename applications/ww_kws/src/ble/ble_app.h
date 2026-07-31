/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef BLE_APP_H_
#define BLE_APP_H_

/**
 * @{
 * @ingroup ww_kws
 */

#if IS_ENABLED(CONFIG_BT)

/**
 * @brief Initialize Bluetooth and start connectable advertising.
 *
 * Advertises the SMP service when @kconfig{CONFIG_MCUMGR_TRANSPORT_BT} is
 * enabled (nRF Connect Device Manager DFU) and/or the Memfault Diagnostic
 * Service when @kconfig{CONFIG_MODELS_OBSERVABILITY_MDS} is enabled.
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
