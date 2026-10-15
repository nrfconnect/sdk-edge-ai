/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/app_version.h>
#include <zephyr/audio/dmic.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#if IS_ENABLED(CONFIG_APP_MODEL_OTA)
#include <zephyr/dfu/mcuboot.h>
#include <zephyr/storage/flash_map.h>
#endif

#include "ble/ble_mds.h"
#include "control_output.h"
#include "dmic.h"
#include "kws/kws.h"
#include "leds.h"
#if IS_ENABLED(CONFIG_APP_MODEL_OTA)
#include "model_update.h"
#endif
#include "ww/wakeword.h"

LOG_MODULE_REGISTER(main);

#define DMIC_READ_TIMEOUT 100

static const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));

#if IS_ENABLED(CONFIG_APP_MODEL_OTA)
static void model_version_print(const char *name, uint8_t area_id)
{
	struct mcuboot_img_header hdr;
	int err;

	err = boot_read_bank_header(area_id, &hdr, sizeof(hdr));
	if (err) {
		LOG_INF("  %s model: not available (err %d)", name, err);
		return;
	}

	LOG_INF("  %s model: %u.%u.%u+%u", name, hdr.h.v1.sem_ver.major, hdr.h.v1.sem_ver.minor,
		hdr.h.v1.sem_ver.revision, hdr.h.v1.sem_ver.build_num);
}
#endif

static void version_banner_print(void)
{
	LOG_INF("Versions:");
	LOG_INF("  application: %s", APP_VERSION_STRING);
#if IS_ENABLED(CONFIG_APP_MODEL_OTA)
	model_version_print("ww", PARTITION_ID(model_storage_ww));
	model_version_print("kws", PARTITION_ID(model_storage_kws));
#else
	LOG_INF("  models: built into application");
#endif
#if IS_ENABLED(CONFIG_WW_KWS_MEMFAULT_FOTA)
	LOG_INF("  Memfault release: %s", CONFIG_MEMFAULT_NCS_FW_VERSION);
#else
	LOG_INF("  Memfault release: disabled");
#endif
}

static int ww_loop(void)
{
	int err;
	void *audio_buffer;
	size_t audio_buffer_size;
	bool ww_detected;

	ww_reset();

	print_control_output((struct control_message){CONTROL_MESSAGE_WAITING_WW});

	while (true) {
#if IS_ENABLED(CONFIG_APP_MODEL_OTA)
		if (model_update_is_pending_reset()) {
			k_sleep(K_FOREVER);
			continue;
		}
#endif
		err = dmic_read(dmic_dev, 0, &audio_buffer, &audio_buffer_size, DMIC_READ_TIMEOUT);
		if (err < 0) {
			LOG_ERR("Failed to read from DMIC (err %d)", err);
			return err;
		}

		err = ww_process(audio_buffer, DMIC_SAMPLES_IN_BLOCK, &ww_detected);
		if (err == -EBUSY) {
			/* More data is needed. */
			continue;
		} else if (err < 0) {
			LOG_ERR("Wakeword detection failed (err %d)", err);
			return err;
		}

		if (ww_detected) {
			print_control_output((struct control_message){CONTROL_MESSAGE_WW_DETECTED});
			if (IS_ENABLED(CONFIG_APP_MODE_WW_ONLY)) {
				leds_blink_led0();
			} else {
				return 0;
			}
		}
	}
}

static int kws_loop(void)
{
	int err;
	void *audio_buffer;
	size_t audio_buffer_size;
	struct kws_prediction prediction;

	uint32_t spotting_timeout = k_uptime_get_32() + CONFIG_KWS_PERIOD_MS;

	kws_reset();

	print_control_output((struct control_message){.type = CONTROL_MESSAGE_WAITING_KW});

	while (IS_ENABLED(CONFIG_APP_MODE_KWS_ONLY) || spotting_timeout > k_uptime_get_32()) {
#if IS_ENABLED(CONFIG_APP_MODEL_OTA)
		if (model_update_is_pending_reset()) {
			k_sleep(K_FOREVER);
			continue;
		}
#endif
		err = dmic_read(dmic_dev, 0, &audio_buffer, &audio_buffer_size, DMIC_READ_TIMEOUT);
		if (err < 0) {
			LOG_ERR("Failed to read from DMIC (err %d)", err);
			return err;
		}

		err = kws_process(audio_buffer, DMIC_SAMPLES_IN_BLOCK, &prediction);
		if (err == -EBUSY) {
			/* More data is needed. */
			continue;
		} else if (err) {
			LOG_ERR("Keyword spotting failed (err %d)", err);
			return err;
		}

		if (prediction.valid) {
			leds_blink_led1();
			spotting_timeout = k_uptime_get_32() + CONFIG_KWS_PERIOD_MS;
			print_control_output(
				(struct control_message){.type = CONTROL_MESSAGE_KW_SPOTTED,
							 .kw_class = prediction.class,
							 .name = prediction.name});
		}
	}

	print_control_output((struct control_message){.type = CONTROL_MESSAGE_TIMEOUT_KWS});

	return 0;
}

int main(void)
{
	int err;

	version_banner_print();

	err = dmic_init();
	if (err) {
		return err;
	}

	err = leds_init();
	if (err) {
		return err;
	}

	err = control_output_init();
	if (err) {
		return err;
	}

	if (IS_ENABLED(CONFIG_APP_MODE_WW_GATED_KWS) || IS_ENABLED(CONFIG_APP_MODE_WW_ONLY)) {
		err = ww_init();
		if (err) {
			return err;
		}
	}

	if (IS_ENABLED(CONFIG_APP_MODE_WW_GATED_KWS) || IS_ENABLED(CONFIG_APP_MODE_KWS_ONLY)) {
		err = kws_init();
		if (err) {
			return err;
		}
	}

	err = init_app_ble();
	if (err) {
		return err;
	}

#if IS_ENABLED(CONFIG_APP_MODEL_OTA)
	err = model_update_init();
	if (err) {
		return err;
	}
#endif

	LOG_INF("Initialization completed, check output on VCOM0");

	err = dmic_trigger(dmic_dev, DMIC_TRIGGER_START);
	if (err < 0) {
		LOG_ERR("Failed to start DMIC (err %d)", err);
		return err;
	}

	while (true) {
		if (IS_ENABLED(CONFIG_APP_MODE_WW_GATED_KWS) ||
		    IS_ENABLED(CONFIG_APP_MODE_WW_ONLY)) {
			err = ww_loop();
			if (err) {
				return err;
			}
		}

		if (IS_ENABLED(CONFIG_APP_MODE_WW_GATED_KWS)) {
			leds_on_led0();
		}

		if (IS_ENABLED(CONFIG_APP_MODE_WW_GATED_KWS) ||
		    IS_ENABLED(CONFIG_APP_MODE_KWS_ONLY)) {
			err = kws_loop();
			if (err) {
				return err;
			}
		}

		if (IS_ENABLED(CONFIG_APP_MODE_WW_GATED_KWS)) {
			leds_off_led0();
		}
	}

	return 0;
}
