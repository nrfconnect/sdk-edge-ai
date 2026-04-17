/*
 * Copyright (c) 2021 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "app_task.h"

#include "app/matter_init.h"
#include "app/task_executor.h"
#include "board/board.h"
#include "lib/core/CHIPError.h"
#include "clusters/identify.h"
#include "switch.h"

#include "nrf_edgeai_task.h"

#include <setup_payload/OnboardingCodesUtil.h>

#include <app-common/zap-generated/attributes/Accessors.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(app, CONFIG_CHIP_APP_LOG_LEVEL);

using namespace ::chip;
using namespace ::chip::app;
using namespace ::chip::app::Clusters;
using namespace ::chip::app::Clusters::OnOff;
using namespace ::chip::DeviceLayer;

#define APPLICATION_BUTTON_MASK DK_BTN2_MSK

namespace
{
Nrf::Matter::IdentifyCluster sIdentifyCluster(Nrf::Matter::GetSwitch().GetSwitchEndpointId());

void ButtonEventHandler(Nrf::ButtonState state, Nrf::ButtonMask hasChanged)
{
	if ((APPLICATION_BUTTON_MASK & state & hasChanged)) {
		Nrf::Matter::GetSwitch().InitiateActionSwitch(::Switch::Action::Toggle);
	}
}
} // namespace

CHIP_ERROR AppTask::Init()
{
	ReturnErrorOnFailure(
		Nrf::Matter::PrepareServer(Nrf::Matter::InitData{.mPostServerInitClbk = [] {
			Nrf::Matter::GetSwitch().Init();
			return CHIP_NO_ERROR;
		}}));

	if (!Nrf::GetBoard().Init(ButtonEventHandler)) {
		LOG_ERR("User interface initialization failed.");
		return CHIP_ERROR_INCORRECT_STATE;
	}

	ReturnErrorOnFailure(
		Nrf::Matter::RegisterEventHandler(Nrf::Board::DefaultMatterEventHandler, 0));

	ReturnErrorOnFailure(sIdentifyCluster.Init());

	return Nrf::Matter::StartServer();
}

CHIP_ERROR AppTask::StartApp()
{
	ReturnErrorOnFailure(Init());

	ReturnErrorOnFailure(EdgeAITask::Instance().Start());
	EdgeAITask::Instance().Enable();

	while (true) {
		Nrf::DispatchNextTask();
	}

	return CHIP_NO_ERROR;
}

void MatterPostAttributeChangeCallback(const chip::app::ConcreteAttributePath &attributePath,
				       uint8_t type, uint16_t size, uint8_t *value)
{
	ClusterId clusterId = attributePath.mClusterId;
	AttributeId attributeId = attributePath.mAttributeId;

	if (clusterId == OnOff::Id && attributeId == OnOff::Attributes::OnOff::Id) {
		LOG_INF("Cluster OnOff: attribute OnOff set to %" PRIu8 "", *value);

		bool value_invert = !*value;

		Nrf::GetBoard().GetLED(Nrf::DeviceLeds::LED2).Set(value_invert);
	}
}
