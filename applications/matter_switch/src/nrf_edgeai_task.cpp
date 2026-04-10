/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/audio/dmic.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/time_units.h>



#include "board/board.h"
#include "app/task_executor.h"

#include "dmic.h"
#include "wakeword.h"
#include "nrf_edgeai_task.h"
#include "app_task.h"

#include <app-common/zap-generated/attributes/Accessors.h>
#include <platform/CHIPDeviceLayer.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(app, CONFIG_CHIP_APP_LOG_LEVEL);

using namespace ::chip;
using namespace ::chip::app;
using namespace ::chip::app::Clusters;
using namespace ::chip::app::Clusters::OnOff;
using namespace ::chip::DeviceLayer;

namespace
{
const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));

void switch_thread_fn()
{
	void *audio_buffer;
	size_t audio_buffer_size;
	bool ww_detected;
	const int32_t read_timeout = 100;
	int err = 0;

	if (dmic_init()) {
		LOG_ERR("Failed to initialize DMIC");
		return;
	}

	if (ww_init() != 0) {
		LOG_ERR("Failed to initialize wakeword detection");
		return;
	}

	LOG_INF("Edge AI Switch initialization completed");

	if (dmic_trigger(dmic_dev, DMIC_TRIGGER_START) < 0) {
		LOG_ERR("Failed to start DMIC");
		return;
	}

	LOG_INF("\n\nWaiting for wakeword...\n\n");

	while (true) {
		err = dmic_read(dmic_dev, 0, &audio_buffer, &audio_buffer_size, read_timeout);
		if (err != 0) {
			continue;
		}

		if (!EdgeAITask::Instance().IsEnabled()) {
			free_dmic_buffer(audio_buffer);
			continue;
		}

		/* ww_process feeds the model and returns the DMIC block to the slab. */
		err = ww_process(reinterpret_cast<uint8_t *>(audio_buffer),
				 audio_buffer_size / DMIC_SAMPLE_BYTES, &ww_detected);
		if (err == -EBUSY) {
			continue;
		} else if (err < 0) {
			LOG_ERR("Wakeword detection failed (err %d)", err);
		}

		if (ww_detected) {
			Nrf::PostTask([] {
				LOG_INF("wakeword detected");
				bool value_invert =
					!Nrf::GetBoard().GetLED(Nrf::DeviceLeds::LED2).GetState();
				Nrf::GetBoard().GetLED(Nrf::DeviceLeds::LED2).Set(value_invert);

				Nrf::Matter::GetSwitch().InitiateActionSwitch(::Switch::Action::Toggle);
			});
		}
	}
}

K_THREAD_DEFINE(switch_thread_id, CONFIG_SWITCH_THREAD_STACK_SIZE, switch_thread_fn, NULL, NULL,
		NULL, CONFIG_SWITCH_THREAD_PRIORITY, K_FP_REGS, SYS_FOREVER_MS);
} // namespace

CHIP_ERROR EdgeAITask::Start()
{
	k_thread_start(switch_thread_id);
	return CHIP_NO_ERROR;
}
