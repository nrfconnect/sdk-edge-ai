/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "dimming_effect.h"

#include "app/task_executor.h"
#include "board/board.h"
#include "pwm/pwm_device.h"

#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/util.h> /* MAX, MIN, ARG_UNUSED */

#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(dimming_effect, LOG_LEVEL_INF);

namespace Nrf
{
namespace DimmingEffect
{
namespace
{

#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)

constexpr uint8_t kPwmMinLevel = 0;
constexpr uint8_t kPwmMaxLevel = 254;

const struct pwm_dt_spec kPwmSpecs[] = {
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led0)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led1)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led2)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led3)),
};

Nrf::PWMDevice s_pwm_devices[ARRAY_SIZE(kPwmSpecs)];
bool s_pwm_channel_ok[ARRAY_SIZE(kPwmSpecs)];

k_timer s_effect_timer;
k_work_delayable s_dim_work;
int s_elapsed_ms;

Config s_cfg;
OnComplete s_on_complete;
void *s_on_complete_ctx;

int blink_pair_count()
{
	return MAX(10, s_cfg.blink_pairs_multiplier * s_cfg.effect_timeout_s);
}

/** @return 0 if at least one channel initialized, -ENODEV if all failed */
int init_pwm_devices_and_turn_on()
{
	unsigned ok = 0;

	for (unsigned i = 0; i < ARRAY_SIZE(s_pwm_devices); i++) {
		Nrf::PWMDevice &dev = s_pwm_devices[i];

		s_pwm_channel_ok[i] = false;
		if (dev.Init(&kPwmSpecs[i], kPwmMinLevel, kPwmMaxLevel, kPwmMaxLevel) != 0) {
			LOG_ERR("PWMDevice init failed for channel %u", i);
			continue;
		}
		dev.SetCallbacks(nullptr, nullptr);
		(void)dev.InitiateAction(Nrf::PWMDevice::ON_ACTION, 0, nullptr);
		s_pwm_channel_ok[i] = true;
		ok++;
	}

	return ok > 0 ? 0 : -ENODEV;
}

void suppress_all_pwm_outputs()
{
	for (unsigned i = 0; i < ARRAY_SIZE(s_pwm_devices); i++) {
		if (s_pwm_channel_ok[i]) {
			s_pwm_devices[i].SuppressOutput();
		}
	}
}

void dim_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);

	const int total_ms = s_cfg.effect_timeout_s * 1000;
	const int num_pairs = blink_pair_count();
	const int fx = s_cfg.fx_tick_ms;
	const int pair_ms = MAX(2 * fx, total_ms / num_pairs);
	const int half_ms = MAX(fx, pair_ms / 2);

	const int e = MIN(s_elapsed_ms, total_ms - 1);
	const int pos = (pair_ms > 0) ? (e % pair_ms) : 0;
	uint8_t level;

	if (pos < half_ms) {
		level = static_cast<uint8_t>(
			(static_cast<uint32_t>(kPwmMaxLevel) * (half_ms - pos)) /
			static_cast<unsigned>(half_ms));
	} else {
		const int up = pair_ms - half_ms;

		if (up < 1) {
			level = kPwmMaxLevel;
		} else {
			const int pos2 = pos - half_ms;

			level = static_cast<uint8_t>((static_cast<uint32_t>(kPwmMaxLevel) *
						      static_cast<unsigned>(pos2)) /
						     static_cast<unsigned>(up));
		}
	}

	for (unsigned i = 0; i < ARRAY_SIZE(s_pwm_devices); i++) {
		if (!s_pwm_channel_ok[i]) {
			continue;
		}
		uint8_t value = level;

		(void)s_pwm_devices[i].InitiateAction(Nrf::PWMDevice::LEVEL_ACTION, 0, &value);
	}

	s_elapsed_ms += fx;
	if (s_elapsed_ms < total_ms) {
		(void)k_work_reschedule(&s_dim_work, K_MSEC(fx));
	} else {
		suppress_all_pwm_outputs();
		Nrf::GetBoard().RunLedStateHandler();
		if (s_on_complete != nullptr) {
			s_on_complete(s_on_complete_ctx);
		}
	}
}

void effect_timer_callback(k_timer *timer)
{
	ARG_UNUSED(timer);
	Nrf::PostTask([] { Start(); });
}

#endif /* CONFIG_PWM && pwm_led aliases */

} // namespace

bool IsAvailable()
{
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)
	return true;
#else
	return false;
#endif
}

int Init(const Config &cfg, OnComplete on_complete, void *on_complete_ctx)
{
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)
	s_cfg = cfg;
	s_on_complete = on_complete;
	s_on_complete_ctx = on_complete_ctx;

	k_timer_init(&s_effect_timer, effect_timer_callback, nullptr);
	k_work_init_delayable(&s_dim_work, dim_work_handler);
	return 0;
#else
	ARG_UNUSED(cfg);
	ARG_UNUSED(on_complete);
	ARG_UNUSED(on_complete_ctx);
	return 0;
#endif
}

void Start()
{
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)

	(void)k_work_cancel_delayable(&s_dim_work);

	s_elapsed_ms = 0;

	Nrf::GetBoard().ForEachLED([](Nrf::LEDWidget &led) { led.Set(false); });

	if (init_pwm_devices_and_turn_on() != 0) {
		LOG_ERR("Dimming effect: no PWM channel initialized");
		Nrf::GetBoard().RunLedStateHandler();
		if (s_on_complete != nullptr) {
			s_on_complete(s_on_complete_ctx);
		}
		return;
	}

	(void)k_work_schedule(&s_dim_work, K_NO_WAIT);
#endif
}

void Stop()
{
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)
	k_timer_stop(&s_effect_timer);
	(void)k_work_cancel_delayable(&s_dim_work);
	suppress_all_pwm_outputs();
	Nrf::GetBoard().RunLedStateHandler();
#endif
}

void RequestStartViaTimer()
{
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)
	k_timer_stop(&s_effect_timer);
	k_timer_start(&s_effect_timer, K_NO_WAIT, K_NO_WAIT);
#endif
}

} // namespace DimmingEffect
} // namespace Nrf
