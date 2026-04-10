/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <platform/CHIPDeviceLayer.h>
#include "switch.h"

class EdgeAITask
{
public:
	static EdgeAITask &Instance()
	{
		static EdgeAITask sEdgeAITask;
		return sEdgeAITask;
	};

	CHIP_ERROR Start();

	void Enable()
	{
		enabled = true;
	}
	void Disable()
	{
		enabled = false;
	}
	bool IsEnabled()
	{
		return enabled;
	}

private:
	bool enabled{false};
};
