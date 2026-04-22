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
#include "dimming_effect.h"

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

#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) && DT_HAS_ALIAS(pwm_led2) &&            \
	DT_HAS_ALIAS(pwm_led3)
	{
		Nrf::DimmingEffect::Config dim_cfg;

		dim_cfg.effect_timeout_s = CONFIG_KW_DETECTION_TIMEOUT_S;
		dim_cfg.blink_pairs_multiplier = 3;
		(void)Nrf::DimmingEffect::Init(dim_cfg);
	}
#endif

	LOG_INF("Edge AI Switch initialization completed");

	if (dmic_trigger(dmic_dev, DMIC_TRIGGER_START) < 0) {
		LOG_ERR("Failed to start DMIC");
		return;
	}

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

		/* ww_process takes the DMIC buffer and returns it to the slab. */
		err = ww_process(reinterpret_cast<uint8_t *>(audio_buffer),
				 audio_buffer_size / DMIC_SAMPLE_BYTES, &ww_detected);
		if (err == -EBUSY) {
			continue;
		}
		if (err < 0) {
			LOG_ERR("Wakeword detection failed (err %d)", err);
			continue;
		}

		if (ww_detected) {
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) && DT_HAS_ALIAS(pwm_led2) &&            \
	DT_HAS_ALIAS(pwm_led3)
			Nrf::DimmingEffect::RequestStartViaTimer();
#endif
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

void EdgeAITask::Enable()
{
	LOG_INF("\n\nWaiting for wakeword...\n\n");
	enabled.store(true, std::memory_order_release);
}

void EdgeAITask::Disable()
{
	enabled.store(false, std::memory_order_release);
	LOG_INF("\n\nEdge AI listening disabled\n\n");
}
