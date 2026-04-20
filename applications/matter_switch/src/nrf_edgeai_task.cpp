/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/audio/dmic.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h>

#include "board/board.h"
#include "app/task_executor.h"
#include "pwm/pwm_device.h"

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

#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) && DT_HAS_ALIAS(pwm_led2) &&            \
	DT_HAS_ALIAS(pwm_led3)

/* Same level range as Matter light_bulb sample (pwm_device maps to duty). */
constexpr uint8_t kKwPwmMinLevel = 0;
constexpr uint8_t kKwPwmMaxLevel = 254;
/** Work queue tick for the wakeword PWM effect (smaller = smoother / more CPU). */
constexpr int kKwFxTickMs = 20;

static const struct pwm_dt_spec s_kw_pwm_specs[] = {
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led0)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led1)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led2)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led3)),
};

static Nrf::PWMDevice s_kw_pwm_devices[ARRAY_SIZE(s_kw_pwm_specs)];

static k_timer s_kw_effect_timer;
static k_work_delayable s_kw_dim_work;
static int s_elapsed_ms;

/** Number of full bright -> dim -> bright cycles packed into the timeout window. */
static int KwBlinkPairCount()
{
	const int s = CONFIG_KW_DETECTION_TIMEOUT_S;

	/* ~3 pairs per second of timeout (e.g. 10 s -> 30 fast “blinks” with dim ramps). */
	return MAX(10, 3 * s);
}

static void KwInitPwmDevicesAndTurnOn()
{
	for (unsigned i = 0; i < ARRAY_SIZE(s_kw_pwm_devices); i++) {
		Nrf::PWMDevice &dev = s_kw_pwm_devices[i];

		if (dev.Init(&s_kw_pwm_specs[i], kKwPwmMinLevel, kKwPwmMaxLevel, kKwPwmMaxLevel) != 0) {
			LOG_ERR("PWMDevice init failed for channel %u", i);
			continue;
		}
		dev.SetCallbacks(nullptr, nullptr);
		(void)dev.InitiateAction(Nrf::PWMDevice::ON_ACTION, 0, nullptr);
	}
}

static void KwSuppressAllPwmOutputs()
{
	for (Nrf::PWMDevice &dev : s_kw_pwm_devices) {
		dev.SuppressOutput();
	}
}

static void KwDimStepWorkHandler(struct k_work *work)
{
	ARG_UNUSED(work);

	const int total_ms = CONFIG_KW_DETECTION_TIMEOUT_S * 1000;
	const int num_pairs = KwBlinkPairCount();
	const int pair_ms = MAX(2 * kKwFxTickMs, total_ms / num_pairs);
	const int half_ms = MAX(kKwFxTickMs, pair_ms / 2);

	const int e = MIN(s_elapsed_ms, total_ms - 1);
	const int pos = (pair_ms > 0) ? (e % pair_ms) : 0;
	uint8_t level;

	if (pos < half_ms) {
		/* Fade down: full brightness toward off. */
		level = static_cast<uint8_t>((static_cast<uint32_t>(kKwPwmMaxLevel) * (half_ms - pos)) /
					      static_cast<unsigned>(half_ms));
	} else {
		/* Fade up: off toward full brightness. */
		const int up = pair_ms - half_ms;

		if (up < 1) {
			level = kKwPwmMaxLevel;
		} else {
			const int pos2 = pos - half_ms;

			level = static_cast<uint8_t>((static_cast<uint32_t>(kKwPwmMaxLevel) *
							static_cast<unsigned>(pos2)) /
						       static_cast<unsigned>(up));
		}
	}

	for (Nrf::PWMDevice &dev : s_kw_pwm_devices) {
		uint8_t value = level;

		(void)dev.InitiateAction(Nrf::PWMDevice::LEVEL_ACTION, 0, &value);
	}

	s_elapsed_ms += kKwFxTickMs;
	if (s_elapsed_ms < total_ms) {
		(void)k_work_reschedule(&s_kw_dim_work, K_MSEC(kKwFxTickMs));
	} else {
		KwSuppressAllPwmOutputs();
		Nrf::GetBoard().RunLedStateHandler();
	}
}

static void KwStartDimmingFromWakewordEffect()
{
	(void)k_work_cancel_delayable(&s_kw_dim_work);

	s_elapsed_ms = 0;

	Nrf::GetBoard().ForEachLED([](Nrf::LEDWidget &led) { led.Set(false); });

	KwInitPwmDevicesAndTurnOn();

	(void)k_work_schedule(&s_kw_dim_work, K_NO_WAIT);
}

static void KwEffectTimerCallback(k_timer *timer)
{
	ARG_UNUSED(timer);
	Nrf::PostTask([] { KwStartDimmingFromWakewordEffect(); });
}

#endif /* CONFIG_PWM && pwm_led DT aliases */

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

#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) && DT_HAS_ALIAS(pwm_led2) &&            \
	DT_HAS_ALIAS(pwm_led3)
	k_timer_init(&s_kw_effect_timer, KwEffectTimerCallback, nullptr);
	k_work_init_delayable(&s_kw_dim_work, KwDimStepWorkHandler);
#endif

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
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) && DT_HAS_ALIAS(pwm_led2) &&            \
	DT_HAS_ALIAS(pwm_led3)
			k_timer_stop(&s_kw_effect_timer);
			k_timer_start(&s_kw_effect_timer, K_NO_WAIT, K_NO_WAIT);
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
