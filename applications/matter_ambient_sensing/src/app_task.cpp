/*
 * Copyright (c) 2021 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "app_task.h"

#include "app/matter_init.h"
#include "app/task_executor.h"
#include "board/board.h"
#include "clusters/identify.h"
#include "lib/core/CHIPError.h"
#include "nrf_edgeai_task.h"

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(app, CONFIG_CHIP_APP_LOG_LEVEL);

using namespace ::chip;
using namespace ::chip::app;
using namespace ::chip::app::Clusters;
using namespace ::chip::app::Clusters::OnOff;
using namespace ::chip::DeviceLayer;

namespace
{
Nrf::Matter::IdentifyCluster sIdentifyCluster(CONFIG_AMBIENT_SENSING_ENDPOINT_ID);

void matter_app_event_handler(const chip::DeviceLayer::ChipDeviceEvent *event, intptr_t arg)
{
	(void)arg;

	switch (event->Type) {
	case chip::DeviceLayer::DeviceEventType::kServerReady:
		// Enable Edge AI task when Matter server is ready
		EdgeAITask::Instance().Enable();
		break;
	case chip::DeviceLayer::DeviceEventType::kCHIPoBLEConnectionEstablished:
		// Disable Edge AI task when commissioning is in progress
		EdgeAITask::Instance().Disable();
		break;
	default:
		break;
	}
}
} // namespace

CHIP_ERROR AppTask::Init()
{
	ReturnErrorOnFailure(Nrf::Matter::PrepareServer(
		Nrf::Matter::InitData{.mPostServerInitClbk = [] { return CHIP_NO_ERROR; }}));

	if (!Nrf::GetBoard().Init()) {
		LOG_ERR("User interface initialization failed.");
		return CHIP_ERROR_INCORRECT_STATE;
	}

	ReturnErrorOnFailure(Nrf::Matter::RegisterEventHandler(matter_app_event_handler, 0));
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
