/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ble_common.h"

#include <zephyr/sys/atomic.h>

static atomic_t ble_connected = ATOMIC_INIT(0);

void ble_common_set_connected(bool connected)
{
	atomic_set(&ble_connected, connected ? 1 : 0);
}

bool ble_common_is_connected(void)
{
	return atomic_get(&ble_connected) != 0;
}

void ble_common_init(void)
{
	atomic_set(&ble_connected, 0);
}
