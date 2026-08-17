/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef __BLE_COMMON_H__
#define __BLE_COMMON_H__

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void ble_common_set_connected(bool connected);
bool ble_common_is_connected(void);
void ble_common_init(void);

#ifdef __cplusplus
}
#endif

#endif /* __BLE_COMMON_H__ */
