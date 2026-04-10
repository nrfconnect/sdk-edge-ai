/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "switch.h"

#if defined(CONFIG_LIGHT_SWITCH)
#include "light_switch.h"
#elif defined(CONFIG_GENERIC_SWITCH)
#include "generic_switch.h"
#endif

#if defined(CONFIG_LIGHT_SWITCH)
LightSwitch sSwitchInstance;
#elif defined(CONFIG_GENERIC_SWITCH)
GenericSwitch sSwitchInstance;
#endif

::Switch &Nrf::Matter::GetSwitch()
{
	return sSwitchInstance;
}
