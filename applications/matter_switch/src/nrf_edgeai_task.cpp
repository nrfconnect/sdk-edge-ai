/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>

#include <zephyr/audio/dmic.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/kernel.h>

#include "board/board.h"
#include "app/task_executor.h"

#include "platform/PlatformManager.h"
#include "pwm/pwm_device.h"
#include "dimming_effect.h"
#include <haly/nrfy_pdm.h>

#include "dmic.h"
#include "wakeword.h"
#include "keyword.h"
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

enum detection_state_e {
	WAITING_FOR_WAKEWORD,
	WAITING_FOR_KEYWORDS,
};

const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));
void switch_thread_fn();
K_THREAD_DEFINE(switch_thread_id, CONFIG_SWITCH_THREAD_STACK_SIZE, switch_thread_fn, NULL, NULL,
		NULL, CONFIG_SWITCH_THREAD_PRIORITY, K_FP_REGS, SYS_FOREVER_MS);
void switch_thread_fn()
{
	void *audio_buffer;
	size_t audio_buffer_size;
	const int32_t read_timeout = 100;
	int err = 0;
	uint32_t kws_start_time = 0;
	detection_state_e app_state = WAITING_FOR_WAKEWORD;
	uint16_t class_detected = KEYWORD_OTHER;

	if (dmic_init()) {
		LOG_ERR("Failed to initialize DMIC");
		return;
	}

	if (ww_init() != 0) {
		LOG_ERR("Failed to initialize wakeword detection");
		return;
	}

	if (kw_init() != 0) {
		LOG_ERR("Failed to initialize keyword detection");
		return;
	}

	if (Nrf::DimmingEffect::IsAvailable()) {
		Nrf::DimmingEffect::Config dim_cfg;

		dim_cfg.effect_timeout_s = CONFIG_KW_DETECTION_TIMEOUT_S;
		dim_cfg.blink_pairs_multiplier = 3;
		(void)Nrf::DimmingEffect::Init(dim_cfg);
	}

	LOG_INF("Edge AI Switch initialization completed, Waiting for Matter to Start");
	k_sem_take(&gMatterStartedSem, K_FOREVER);
	LOG_INF("Matter server is ready, starting Edge AI capture");

	if (dmic_trigger(dmic_dev, DMIC_TRIGGER_START) < 0) {
		LOG_ERR("Failed to start DMIC");
		return;
	}
	nrfy_pdm_gain_set(NRF_PDM20, 0x40, 0x40);

	LOG_INF("\n\nWaiting for wakeword...\n\n");

	while (true) {
		bool ww_detected = false;

		err = dmic_read(dmic_dev, 0, &audio_buffer, &audio_buffer_size, read_timeout);
		if (err != 0) {
			if (err == -EAGAIN || err == -EBUSY) {
				k_yield();
				continue;
			}
			LOG_WRN("DMIC read failed (err %d)", err);
			k_sleep(K_MSEC(1));
			continue;
		}

		if (!EdgeAITask::Instance().IsEnabled()) {
			free_dmic_buffer(audio_buffer);
			continue;
		}
		switch (app_state) {
		case WAITING_FOR_WAKEWORD: {
			/* ww_process feeds the model and returns the DMIC block to the slab. */
			err = ww_process(reinterpret_cast<uint8_t *>(audio_buffer),
					 audio_buffer_size / DMIC_SAMPLE_BYTES, &ww_detected);
			if (err == -EBUSY) {
				continue;
			} else if (err < 0) {
				LOG_ERR("Wakeword detection failed (err %d)", err);
			}

			if (ww_detected) {

				LOG_INF("wakeword detected, Looking for keywords");
				kw_reset_model();
				kws_start_time = k_uptime_get_32();
				app_state = WAITING_FOR_KEYWORDS;

				if (Nrf::DimmingEffect::IsAvailable()) {
					Nrf::DimmingEffect::RequestStartViaTimer();
				}
			}
			break;
		}
		case WAITING_FOR_KEYWORDS: {
			if (k_uptime_get_32() - kws_start_time >
			    CONFIG_KW_DETECTION_TIMEOUT_S * 1000) {
				ww_reset_model();
				app_state = WAITING_FOR_WAKEWORD;
				LOG_INF("\n\nWaiting for wakeword...\n\n");
				break;
			}

			const int kws =
				kw_process(reinterpret_cast<uint8_t *>(audio_buffer),
					   audio_buffer_size / DMIC_SAMPLE_BYTES, &class_detected);
			if (kws == -EBUSY) {
				continue;
			}
			if (kws < 0) {
				LOG_ERR("Keyword detection failed (err %d)", kws);
				break;
			}
			/* 0: keep listening; 1: full phrase recognized. */
			if (kws == 0) {
				break;
			}
			switch (class_detected) {
			case KEYWORD_OFF: {
				SystemLayer().ScheduleLambda([] {
					LOG_INF("Turning light off");
					Nrf::Matter::GetSwitch().InitiateActionSwitch(
						::Switch::Action::Off);
				});
				break;
			}
			case KEYWORD_ON: {
				SystemLayer().ScheduleLambda([] {
					LOG_INF("Turning light on");
					Nrf::Matter::GetSwitch().InitiateActionSwitch(
						::Switch::Action::On);
				});
				break;
			}
			case KEYWORD_SWITCH: {
				SystemLayer().ScheduleLambda([] {
					LOG_INF("Toggling the light");

					Nrf::Matter::GetSwitch().InitiateActionSwitch(
						::Switch::Action::Toggle);
				});
				break;
			}
			default: {
				LOG_DBG("Not a valid keyword (class %u)", class_detected);
			}
			}
			// k_thread_suspend(switch_thread_id);
			ww_reset_model();
			app_state = WAITING_FOR_WAKEWORD;
			LOG_INF("\n\nWaiting for wakeword...\n\n");
			Nrf::DimmingEffect::Stop();
			break;
		}
		}
	}
}

} // namespace

CHIP_ERROR EdgeAITask::Start()
{
	k_thread_start(switch_thread_id);
	return CHIP_NO_ERROR;
}

void EdgeAITask::Enable()
{
	LOG_INF("\n\nAI task Enabled\n\n");
	enabled.store(true, std::memory_order_release);
}

void EdgeAITask::Disable()
{
	enabled.store(false, std::memory_order_release);
	LOG_INF("\n\nEdge AI listening disabled\n\n");
}
