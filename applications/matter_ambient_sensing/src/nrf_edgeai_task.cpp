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
#include "nrf_edgeai_task.h"
#include "app_task.h"
#include "ambient_sensing.h"
#include "models/all_models.h"

#include <app-common/zap-generated/attributes/Accessors.h>
#include <app-common/zap-generated/cluster-objects.h>
#include <app-common/zap-generated/ids/Clusters.h>
#include <app/EventLogging.h>
#include <platform/CHIPDeviceLayer.h>

#include <zephyr/logging/log.h>

LOG_MODULE_DECLARE(app, CONFIG_CHIP_APP_LOG_LEVEL);

using namespace ::chip;
using namespace ::chip::app;
using namespace ::chip::app::Clusters;
using namespace ::chip::DeviceLayer;

namespace
{
const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));

#ifdef CONFIG_USE_OCCUPANCY_SENSOR_INSTEAD_OF_AMBIENT_SENSING
void ApplyOccupancyValue(EndpointId endpointId, uint8_t newOccupancyValue)
{
	using chip::BitMask;
	using chip::Protocols::InteractionModel::Status;

	BitMask<Clusters::OccupancySensing::OccupancyBitmap> currentOccupancy;
	Status status = Clusters::OccupancySensing::Attributes::Occupancy::Get(endpointId,
									       &currentOccupancy);

	if (status != Status::Success) {
		LOG_ERR("Occupancy::Get failed: %u", static_cast<unsigned>(to_underlying(status)));
		return;
	}

	if (static_cast<BitMask<Clusters::OccupancySensing::OccupancyBitmap>>(newOccupancyValue) ==
	    currentOccupancy) {
		return;
	}

	status = Clusters::OccupancySensing::Attributes::Occupancy::Set(endpointId,
									newOccupancyValue);
	if (status != Status::Success) {
		LOG_ERR("Occupancy::Set failed: %u", static_cast<unsigned>(to_underlying(status)));
		return;
	}

	if (newOccupancyValue == 1) {
		LOG_INF("Occupancy is now occupied");
	} else {
		LOG_INF("Occupancy is now vacant");
	}
}

void OccupancySensingChipWorkerHandler(intptr_t arg)
{
	ApplyOccupancyValue(CONFIG_AMBIENT_SENSING_ENDPOINT_ID, static_cast<uint8_t>(arg));
}

CHIP_ERROR RequestOccupancyMatterUpdate(uint8_t occupancyRaw)
{
	return DeviceLayer::PlatformMgr().ScheduleWork(OccupancySensingChipWorkerHandler,
						       static_cast<intptr_t>(occupancyRaw));
}
#endif

void ai_thread_fn()
{
	void *audio_buffer;
	size_t audio_buffer_size;
	const int32_t read_timeout = 100;
	int err = 0;

	LOG_INF("Starting Ambient Sensing Application...");
	LOG_INF("Model postprocessing adjustable parameters:");

	nrf_edgeai_rt_version_t libver = nrf_edgeai_runtime_version();
	LOG_INF("Nordic Edge AI Library version: %d.%d.%d", libver.field.major, libver.field.minor,
		libver.field.patch);

	nrf_edgeai_t *p_model = get_ambient_sensing_model();

	// Initialize ambient sensing model
	nrf_edgeai_err_t res = nrf_edgeai_init(p_model);
	if (res != NRF_EDGEAI_ERR_SUCCESS) {
		LOG_ERR("Failed to initialize Edge AI model %s, error code: %d",
			Nrf::AmbientSensing::getModelName(), res);
		return;
	}

	if (dmic_init()) {
		LOG_ERR("Failed to initialize DMIC");
		return;
	}

#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)
	{
		Nrf::DimmingEffect::Config dim_cfg;

		dim_cfg.effect_timeout_s = 5;
		dim_cfg.blink_pairs_multiplier = 3;
		(void)Nrf::DimmingEffect::Init(
			dim_cfg,
			[](void *) {
#ifdef CONFIG_USE_OCCUPANCY_SENSOR_INSTEAD_OF_AMBIENT_SENSING
				if (RequestOccupancyMatterUpdate(0) != CHIP_NO_ERROR) {
					LOG_ERR("Failed to schedule occupancy clear to Matter "
						"stack");
				}
#endif
			},
			nullptr);
	}
#endif

	LOG_INF("Edge AI initialization completed");

	if (dmic_trigger(dmic_dev, DMIC_TRIGGER_START) < 0) {
		LOG_ERR("Failed to start DMIC");
		return;
	}

	while (true) {
		err = dmic_read(dmic_dev, 0, &audio_buffer, &audio_buffer_size, read_timeout);
		if (err != 0) {
			/* No data yet or driver busy: avoid a tight loop and log spam. */
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
			LOG_INF("%s detected", Nrf::AmbientSensing::getModelName());
			Nrf::PostTask([] {
#if defined(CONFIG_PWM) && DT_HAS_ALIAS(pwm_led0) && DT_HAS_ALIAS(pwm_led1) &&                     \
	DT_HAS_ALIAS(pwm_led2) && DT_HAS_ALIAS(pwm_led3)
				Nrf::DimmingEffect::Start();
#endif
			});
#ifdef CONFIG_USE_OCCUPANCY_SENSOR_INSTEAD_OF_AMBIENT_SENSING
			if (RequestOccupancyMatterUpdate(1) != CHIP_NO_ERROR) {
				LOG_ERR("Failed to schedule occupancy update to Matter stack");
			}
#endif
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

void EdgeAITask::Enable()
{
	LOG_INF("\n\nWaiting for %s source...\n\n", Nrf::AmbientSensing::getModelName());
	enabled.store(true, std::memory_order_release);
}

void EdgeAITask::Disable()
{
	enabled.store(false, std::memory_order_release);
	LOG_INF("\n\nEdge AI listening disabled\n\n");
}

void MatterPostAttributeChangeCallback(const chip::app::ConcreteAttributePath &attributePath,
				       uint8_t type, uint16_t size, uint8_t *value)
{
	ClusterId clusterId = attributePath.mClusterId;
	AttributeId attributeId = attributePath.mAttributeId;

	if (clusterId == OccupancySensing::Id &&
	    attributeId == OccupancySensing::Attributes::Occupancy::Id) {
	}
}
