/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "control_output.h"
#include "dmic.h"
#include "leds.h"
#include "wakeword.h"

#include <stddef.h>
#include <stdint.h>

#include <zephyr/audio/dmic.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(main);

static const struct device *const dmic_dev = DEVICE_DT_GET(DT_NODELABEL(dmic_dev));

int main(void)
{
	int err;

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

	err = ww_init();
	if (err) {
		return err;
	}

	err = dmic_trigger(dmic_dev, DMIC_TRIGGER_START);
	if (err < 0) {
		LOG_ERR("Failed to start DMIC (err %d)", err);
		return err;
	}

	void *audio_buffer;
	uint32_t audio_buffer_size;
	bool ww_detected;

	print_control_output((struct control_message){CONTROL_MESSAGE_WAITING_WW});

	while (true) {
		const int32_t read_timeout = 100;

		err = dmic_read(dmic_dev, 0, &audio_buffer, &audio_buffer_size, read_timeout);
		if (err < 0) {
			LOG_ERR("Failed to read from DMIC (err %d)", err);
			return err;
		}

		err = ww_process(audio_buffer, audio_buffer_size / DMIC_SAMPLE_BYTES, &ww_detected);
		if (err == -EBUSY) {
			/* More data is needed. */
			continue;
		} else if (err < 0) {
			LOG_ERR("Wakeword detection failed (err %d)", err);
			return err;
		}

		if (ww_detected) {
			leds_blink_led0();
			print_control_output((struct control_message){CONTROL_MESSAGE_WW_DETECTED});
		}
	}

	return 0;
}
