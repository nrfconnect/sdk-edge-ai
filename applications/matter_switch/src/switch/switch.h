/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <app/util/basic-types.h>
#include <lib/core/CHIPError.h>

#include "binding/binding_handler.h"
#include "app_task.h"

#include <atomic>

/**
 * @class Switch
 * @brief Base class for all switch types
 *
 * This class provides a base interface for all switch types.
 */
class Switch
{
public:
	enum class Action : uint8_t {
		Toggle, /* Switch state on lighting-app device */
		On,	/* Turn on light on lighting-app device */
		Off,	/* Turn off light on lighting-app device */
		ShortPress, /* Short press on switch */
		LongPress, /* Long press on switch */
	};

	/**
	 * @brief Initialize the switch
	 *
	 * This method will initialize the switch.
	 */
	virtual void Init() = 0;

	/**
	 * @brief Initiate an action on the switch
	 *
	 * This method will initiate an action on the switch.
	 */
	virtual void InitiateActionSwitch(Action action) = 0;

	/**
	 * @brief Change the brightness of the switch
	 *
	 * This method will change the brightness of the switch.
	 */
	virtual void DimmerChangeBrightness() = 0;

	chip::EndpointId GetSwitchEndpointId()
	{
		return sSwitchEndpoint;
	}

protected:
	bool mCurrentState = false;

	constexpr static chip::EndpointId sSwitchEndpoint =
		static_cast<chip::EndpointId>(CONFIG_SWITCH_ENDPOINT_ID);
};

namespace Nrf
{
namespace Matter
{

Switch &GetSwitch();

}
} // namespace Nrf