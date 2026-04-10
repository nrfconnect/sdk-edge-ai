/*
 * Copyright (c) 2022 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once
#include <app/util/basic-types.h>
#include <lib/core/CHIPError.h>

#include "binding/binding_handler.h"
#include "switch.h"

#include <atomic>

/** @class LightSwitch
 *  @brief Class for controlling a CHIP light bulb over a Thread network
 *
 *  Features:
 *  - discovering a CHIP light bulb which advertises itself by sending Thread multicast packets
 *  - toggling and dimming the connected CHIP light bulb by sending appropriate CHIP messages
 */
class LightSwitch : public Switch {
public:
	LightSwitch() = default;

	void Init() override;
	void InitiateActionSwitch(Action action) override;
	void DimmerChangeBrightness() override;

private:
	static void OnOffProcessCommand(chip::CommandId commandId,
					const chip::app::Clusters::Binding::TableEntry &binding,
					chip::OperationalDeviceProxy *device,
					Nrf::Matter::BindingHandler::BindingData &bindingData);

	static void LevelControlProcessCommand(chip::CommandId commandId,
					       const chip::app::Clusters::Binding::TableEntry &binding,
					       chip::OperationalDeviceProxy *device,
					       Nrf::Matter::BindingHandler::BindingData &bindingData);

	static void SwitchChangedHandler(const chip::app::Clusters::Binding::TableEntry &binding,
					 chip::OperationalDeviceProxy *deviceProxy,
					 Nrf::Matter::BindingHandler::BindingData &bindingData);

	constexpr static auto kOnePercentBrightnessApproximation = 3;
	constexpr static auto kMaximumBrightness = 254;
};
