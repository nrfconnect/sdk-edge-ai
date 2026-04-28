/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#include <atomic>

#include <platform/CHIPDeviceLayer.h>
#include "switch.h"

class EdgeAITask
{
      public:
	/**
	 * @brief Get the singleton instance of EdgeAITask.
	 *
	 * @return Reference to the EdgeAITask instance.
	 */
	static EdgeAITask &Instance()
	{
		static EdgeAITask sEdgeAITask;
		return sEdgeAITask;
	};

	/**
	 * @brief Start the Edge AI task thread.
	 *
	 * @return CHIP_ERROR indicating success or failure.
	 */
	CHIP_ERROR Start();

	/**
	 * @brief Enable the Edge AI task.
	 */
	void Enable();

	/**
	 * @brief Disable the Edge AI task.
	 */
	void Disable();

	/**
	 * @brief Check if the Edge AI task is enabled.
	 *
	 * @return true if enabled, false otherwise.
	 */
	bool IsEnabled() const
	{
		return enabled.load(std::memory_order_acquire);
	}

      private:
	/** @brief Flag indicating whether the Edge AI task is enabled (Matter thread vs. Edge AI
	 * thread). */
	std::atomic<bool> enabled{false};
};
