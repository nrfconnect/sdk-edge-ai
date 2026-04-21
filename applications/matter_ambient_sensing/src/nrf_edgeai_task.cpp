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
#include "nrf_edgeai_task.h"
#include "app_task.h"
#include "ambient_sensing.h"
#include "models/all_models.h"

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

#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)

#define DIMMING_TIMEOUT_S 5
#define DIMMING_COUNT	  3

/* Same level range as Matter light_bulb sample (pwm_device maps to duty). */
constexpr uint8_t pwmMinLevel = 0;
constexpr uint8_t pwmMaxLevel = 254;
/** Work queue tick for the wakeword PWM effect (smaller = smoother / more CPU). */
constexpr int fxTickMs = 20;

static const struct pwm_dt_spec pwmSpecs[] = {
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led0)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led1)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led2)),
	PWM_DT_SPEC_GET(DT_ALIAS(pwm_led3)),
};

static Nrf::PWMDevice pwmDevices[ARRAY_SIZE(pwmSpecs)];

static k_timer effectTimer;
static k_work_delayable dimmingWork;
static int elapsedMs;

/** Number of full bright -> dim -> bright cycles packed into the timeout window. */
static int blinkPairCount()
{
	return MAX(10, DIMMING_COUNT * DIMMING_TIMEOUT_S);
}

static void initPWMDevicesAndTurnOn()
{
	for (unsigned i = 0; i < ARRAY_SIZE(pwmDevices); i++) {
		Nrf::PWMDevice &dev = pwmDevices[i];

		if (dev.Init(&pwmSpecs[i], pwmMinLevel, pwmMaxLevel, pwmMaxLevel) != 0) {
			LOG_ERR("PWMDevice init failed for channel %u", i);
			continue;
		}
		dev.SetCallbacks(nullptr, nullptr);
		(void)dev.InitiateAction(Nrf::PWMDevice::ON_ACTION, 0, nullptr);
	}
}

static void suppressAllPWMOutputs()
{
	for (Nrf::PWMDevice &dev : pwmDevices) {
		dev.SuppressOutput();
	}
}

static void dimmingWorkHandler(struct k_work *work)
{
	ARG_UNUSED(work);

	const int total_ms = DIMMING_TIMEOUT_S * 1000;
	const int num_pairs = blinkPairCount();
	const int pair_ms = MAX(2 * fxTickMs, total_ms / num_pairs);
	const int half_ms = MAX(fxTickMs, pair_ms / 2);

	const int e = MIN(elapsedMs, total_ms - 1);
	const int pos = (pair_ms > 0) ? (e % pair_ms) : 0;
	uint8_t level;

	if (pos < half_ms) {
		/* Fade down: full brightness toward off. */
		level = static_cast<uint8_t>(
			(static_cast<uint32_t>(pwmMaxLevel) * (half_ms - pos)) /
			static_cast<unsigned>(half_ms));
	} else {
		/* Fade up: off toward full brightness. */
		const int up = pair_ms - half_ms;

		if (up < 1) {
			level = pwmMaxLevel;
		} else {
			const int pos2 = pos - half_ms;

			level = static_cast<uint8_t>(
				(static_cast<uint32_t>(pwmMaxLevel) * static_cast<unsigned>(pos2)) /
				static_cast<unsigned>(up));
		}
	}

	for (Nrf::PWMDevice &dev : pwmDevices) {
		uint8_t value = level;

		(void)dev.InitiateAction(Nrf::PWMDevice::LEVEL_ACTION, 0, &value);
	}

	elapsedMs += fxTickMs;
	if (elapsedMs < total_ms) {
		(void)k_work_reschedule(&dimmingWork, K_MSEC(fxTickMs));
	} else {
		suppressAllPWMOutputs();
		Nrf::GetBoard().RunLedStateHandler();
	}
}

static void startDimmingEffect()
{
	(void)k_work_cancel_delayable(&dimmingWork);

	elapsedMs = 0;

	Nrf::GetBoard().ForEachLED([](Nrf::LEDWidget &led) { led.Set(false); });

	initPWMDevicesAndTurnOn();

	(void)k_work_schedule(&dimmingWork, K_NO_WAIT);
}

static void effectTimerCallback(k_timer *timer)
{
	ARG_UNUSED(timer);
	Nrf::PostTask([] { startDimmingEffect(); });
}

#endif /* CONFIG_PWM && pwm_led DT aliases */

void ai_thread_fn()
{
	void *audio_buffer;
	size_t audio_buffer_size;
	const int32_t read_timeout = 100;
	int err = 0;

	LOG_INF("Starting Ambient Sensing Application...");
	LOG_INF("Model postprocessig adjustable parameters:");

	nrf_edgeai_rt_version_t libver = nrf_edgeai_runtime_version();
	LOG_INF("Nordic Edge AI Library version: %d.%d.%d", libver.field.major, libver.field.minor,
		libver.field.patch);

	nrf_edgeai_t *p_model = get_ambient_sensing_model();

	// Initialize ambient sensing model
	nrf_edgeai_err_t res = nrf_edgeai_init(p_model);
	if (res != NRF_EDGEAI_ERR_SUCCESS) {
		LOG_ERR("Failed to initialize Edge AI model, error code: %d", res);
		return;
	}

	if (dmic_init()) {
		LOG_ERR("Failed to initialize DMIC");
		return;
	}

#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)
	k_timer_init(&effectTimer, effectTimerCallback, nullptr);
	k_work_init_delayable(&dimmingWork, dimmingWorkHandler);
#endif

	LOG_INF("Edge AI initialization completed");

	if (dmic_trigger(dmic_dev, DMIC_TRIGGER_START) < 0) {
		LOG_ERR("Failed to start DMIC");
		return;
	}

	LOG_INF("\n\nWaiting for ambient sensing source...\n\n");

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

		if (err == -EBUSY) {
			continue;
		} else if (err < 0) {
			LOG_ERR("Ambient sensing source detection failed (err %d)", err);
		}

		size_t samples_num = audio_buffer_size / DMIC_SAMPLE_BYTES;

		// Feed audio data to the model dsp pipeline and
		// waiting for internal buffers to be filled with enough data for feature extraction
		res = nrf_edgeai_feed_inputs(p_model, audio_buffer, samples_num);
		free_dmic_buffer(audio_buffer);

		if (res != NRF_EDGEAI_ERR_SUCCESS) {
			continue;
		}

		// Run feature extraction and model inference
		res = nrf_edgeai_run_inference(p_model);

		if (res != NRF_EDGEAI_ERR_SUCCESS) {
			continue;
		}

		// Run postprocessing on the model output to determine if snoring is detected based
		// on the model inference results
		if (Nrf::AmbientSensing::process(p_model)) {
			printk("Sound detected! \n");
			Nrf::PostTask([] { startDimmingEffect(); });
		}
	}
}

K_THREAD_DEFINE(ai_thread_id, CONFIG_AI_THREAD_STACK_SIZE, ai_thread_fn, NULL, NULL, NULL,
		CONFIG_AI_THREAD_PRIORITY, K_FP_REGS, SYS_FOREVER_MS);
} // namespace

CHIP_ERROR EdgeAITask::Start()
{
	k_thread_start(ai_thread_id);
	return CHIP_NO_ERROR;
}
