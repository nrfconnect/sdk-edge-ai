/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <errno.h>
#include <string.h>

#include <hw_id.h>
#include <zephyr/logging/log.h>
#include <zephyr/settings/settings.h>

#include "memfault_fota.h"

LOG_MODULE_REGISTER(memfault_fota, CONFIG_LOG_DEFAULT_LEVEL);

#define SERIAL_NUMBER_SETTING_KEY "bt/dis/serial"

int memfault_fota_init(void)
{
	char serial_buf[HW_ID_LEN + 1];
	ssize_t serial_len;
	int err;

	serial_len = settings_load_one(SERIAL_NUMBER_SETTING_KEY, serial_buf,
					 sizeof(serial_buf));
	if (serial_len > 0) {
		serial_buf[serial_len] = '\0';
		LOG_INF("Bluetooth DIS serial loaded from settings: %s", serial_buf);
		return 0;
	}

	err = hw_id_get(serial_buf, sizeof(serial_buf));
	if (err != 0) {
		LOG_ERR("Failed to read HW ID for DIS serial (err %d)", err);
		return err;
	}

	serial_len = (ssize_t)strlen(serial_buf);
	err = settings_save_one(SERIAL_NUMBER_SETTING_KEY, serial_buf, (size_t)serial_len);
	if (err != 0) {
		LOG_ERR("Failed to persist DIS serial (err %d)", err);
		return err;
	}

	err = settings_runtime_set(SERIAL_NUMBER_SETTING_KEY, serial_buf, (size_t)serial_len);
	if (err != 0) {
		LOG_ERR("Failed to apply DIS serial at runtime (err %d)", err);
		return err;
	}

	LOG_INF("Bluetooth DIS serial set from HW ID: %s", serial_buf);

	return 0;
}
