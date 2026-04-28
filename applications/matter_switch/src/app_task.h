/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <platform/CHIPDeviceLayer.h>
#include <zephyr/kernel.h>
#include "lib/support/CodeUtils.h"

extern struct k_sem gMatterStartedSem;

class AppTask
{
      public:
	static AppTask &Instance()
	{
		static AppTask sAppTask;
		return sAppTask;
	};

	CHIP_ERROR StartApp();

      private:
	CHIP_ERROR Init();
};
