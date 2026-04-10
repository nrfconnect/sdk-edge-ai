/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "generic_switch.h"

#include <app-common/zap-generated/attributes/Accessors.h>
#include <app-common/zap-generated/cluster-objects.h>
#include <app/clusters/switch-server/switch-server.h>
#include <app/server/Server.h>
#include <platform/CHIPDeviceLayer.h>

#include <zephyr/logging/log.h>

using namespace chip;
using namespace chip::app;
using namespace chip::app::Clusters;
using namespace chip::DeviceLayer;

LOG_MODULE_DECLARE(app, CONFIG_CHIP_APP_LOG_LEVEL);

void GenericSwitch::Init()
{
	// Nothing for now
}

void GenericSwitch::InitiateActionSwitch(Action action)
{
	switch (action) {
	case Action::Toggle:
		// Simulate a complete short press/release cycle for toggle
		if (mCurrentState) {
			InitialPressHandler();
			ShortReleaseHandler();
		} else {
			LongPressHandler();
			LongReleaseHandler();
		}
		mCurrentState = !mCurrentState;
		break;
	case Action::ShortPress:
		InitialPressHandler();
		ShortReleaseHandler();
		break;
	case Action::LongPress:
	LongPressHandler();
		LongReleaseHandler();
		break;
	default:
		break;
	}
}

void GenericSwitch::DimmerChangeBrightness()
{
	// Use long press handler for brightness changes
	// Each call represents a brightness step during dimming
	LongPressHandler();
}

void GenericSwitch::InitialPressHandler()
{
	// Press moves Position from 0 (idle) to 1 (press)
	uint8_t newPosition = 1;

	LOG_DBG("GenericSwitchInitialPress new position %d", newPosition);
	SystemLayer().ScheduleLambda([newPosition, this] {
		Clusters::Switch::Attributes::CurrentPosition::Set(
			Nrf::Matter::GetSwitch().GetSwitchEndpointId(), newPosition);
		// InitialPress event takes newPosition as event data
		SwitchServer::Instance().OnInitialPress(
			Nrf::Matter::GetSwitch().GetSwitchEndpointId(), newPosition);
	});
}

void GenericSwitch::ShortReleaseHandler()
{
	// Release moves Position from 1 (press) to 0 (idle)
	uint8_t previousPosition = 1;
	uint8_t newPosition = 0;

	LOG_DBG("GenericSwitchShortRelease new position %d", newPosition);
	SystemLayer().ScheduleLambda([newPosition, previousPosition, this] {
		Clusters::Switch::Attributes::CurrentPosition::Set(
			Nrf::Matter::GetSwitch().GetSwitchEndpointId(), newPosition);
		// Short Release event takes newPosition as event data
		SwitchServer::Instance().OnShortRelease(
			Nrf::Matter::GetSwitch().GetSwitchEndpointId(), previousPosition);
	});
}

void GenericSwitch::LongReleaseHandler()
{
	// Long release moves Position from 1 (press) to 0 (idle)
	uint8_t previousPosition = 1;
	uint8_t newPosition = 0;

	LOG_DBG("GenericSwitchLongRelease new position %d", newPosition);
	SystemLayer().ScheduleLambda([newPosition, previousPosition, this] {
		Clusters::Switch::Attributes::CurrentPosition::Set(
			Nrf::Matter::GetSwitch().GetSwitchEndpointId(), newPosition);
		// Long Release event takes newPosition as event data
		SwitchServer::Instance().OnLongRelease(
			Nrf::Matter::GetSwitch().GetSwitchEndpointId(), previousPosition);
	});
}

void GenericSwitch::LongPressHandler()
{
	// Long press moves Position from 0 (idle) to 1 (press)
	uint8_t newPosition = 1;

	LOG_DBG("GenericSwitchLongPress new position %d", newPosition);
	SystemLayer().ScheduleLambda([newPosition, this] {
		// LongPress event takes newPosition as event data
		SwitchServer::Instance().OnLongPress(Nrf::Matter::GetSwitch().GetSwitchEndpointId(),
						     newPosition);
	});
}
