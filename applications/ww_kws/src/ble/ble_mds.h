/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef BLE_MDS_H_
#define BLE_MDS_H_

/**
 * @{
 * @ingroup ww_kws
 */

#if IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_MDS)

/**
 * @brief Initialize BLE and start MDS advertising.
 *
 * @return 0 on success, -errno on failure.
 */
int init_app_ble(void);

#else

static inline int init_app_ble(void)
{
	return 0;
}

#endif /* IS_ENABLED(CONFIG_MODELS_OBSERVABILITY_MDS) */

/**
 * @}
 */

#endif /* BLE_MDS_H_ */
