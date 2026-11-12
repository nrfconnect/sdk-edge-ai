/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "accel.h"

#include <errno.h>

#include <zephyr/types.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(accel, LOG_LEVEL_INF);

static const struct device *const accel_dev = DEVICE_DT_GET_ONE(bosch_bmi270);

static accel_data_ready_cb_t accel_data_ready_cb;

static void data_ready_timer_handler(struct k_timer *timer)
{
	ARG_UNUSED(timer);
	if (accel_data_ready_cb) {
		accel_data_ready_cb();
	}
}

static K_TIMER_DEFINE(data_ready_timer, data_ready_timer_handler, NULL);

static int accel_set_attr(enum sensor_attribute attr, int32_t val1)
{
	const struct sensor_value val = {.val1 = val1, .val2 = 0};

	return sensor_attr_set(accel_dev, SENSOR_CHAN_ACCEL_XYZ, attr, &val);
}

int accel_init(accel_data_ready_cb_t data_ready_cb)
{
	int err;

	if (!device_is_ready(accel_dev)) {
		LOG_ERR("Accelerometer device %s is not ready", accel_dev->name);
		return -ENODEV;
	}

	err = accel_set_attr(SENSOR_ATTR_FULL_SCALE, ACCEL_SCALE_G);
	if (err) {
		LOG_ERR("Failed to set full scale to %d G (err %d)", ACCEL_SCALE_G, err);
		return err;
	}

	err = accel_set_attr(SENSOR_ATTR_OVERSAMPLING, 1);
	if (err) {
		LOG_ERR("Failed to set oversampling (err %d)", err);
		return err;
	}

	err = accel_set_attr(SENSOR_ATTR_SAMPLING_FREQUENCY, ACCEL_SAMPLE_RATE_HZ);
	if (err) {
		LOG_ERR("Failed to set sampling frequency to %u Hz (err %d)", ACCEL_SAMPLE_RATE_HZ,
			err);
		return err;
	}

	accel_data_ready_cb = data_ready_cb;
	k_timer_start(&data_ready_timer, K_NO_WAIT, K_NSEC(NSEC_PER_SEC / ACCEL_SAMPLE_RATE_HZ));

	return 0;
}

int accel_read(float p_data[ACCEL_AXES_NUM])
{
	struct sensor_value accel[ACCEL_AXES_NUM];
	int err;

	if (p_data == NULL) {
		LOG_ERR("Output buffer is NULL");
		return -EINVAL;
	}

	/* BMI270 driver supports SENSOR_CHAN_ALL only */
	err = sensor_sample_fetch(accel_dev);
	if (err) {
		return err;
	}

	err = sensor_channel_get(accel_dev, SENSOR_CHAN_ACCEL_XYZ, accel);
	if (err) {
		return err;
	}

	for (uint32_t i = 0; i < ACCEL_AXES_NUM; i++) {
		p_data[i] = sensor_value_to_float(&accel[i]);
	}

	return 0;
}
