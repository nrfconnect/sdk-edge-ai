/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/drivers/sensor.h>

#include "data_fwd_sensor.h"
#include "bme688_common.h"

#define SENSOR_CHANNEL_COUNT BME688_CHANNEL_COUNT
#define SENSOR_TYPE_VALUE    3
#define FREQUENCY_HZ	     BME688_FREQUENCY_HZ

BUILD_ASSERT(FREQUENCY_HZ > 0, "Sampling frequency must be non-zero");

static const char *const channel_names[SENSOR_CHANNEL_COUNT] = {BME688_CHANNEL_NAMES};

static const struct device *dev_bme = DEVICE_DT_GET(BME688_NODE);

static K_SEM_DEFINE(fetch_sem, 0, 1);
static void fetch_timer_handler(struct k_timer *timer);
static K_TIMER_DEFINE(fetch_timer, fetch_timer_handler, NULL);

static void fetch_timer_handler(struct k_timer *timer)
{
	ARG_UNUSED(timer);
	k_sem_give(&fetch_sem);
}

int data_fwd_sensor_init(void)
{
	if (!device_is_ready(dev_bme)) {
		return -ENODEV;
	}

	const uint32_t period_ns = Z_HZ_ns / FREQUENCY_HZ;

	k_timer_start(&fetch_timer, K_NO_WAIT, K_NSEC(period_ns));

	return 0;
}

int data_fwd_sensor_fetch(proto_value_t *values, const size_t values_size, size_t *count)
{
	int err;

	if ((values == NULL) || (count == NULL) || (values_size < SENSOR_CHANNEL_COUNT)) {
		return -EINVAL;
	}

	k_sem_take(&fetch_sem, K_FOREVER);

	err = data_fwd_bme688_fetch(dev_bme);
	if (err) {
		return err;
	}

	*count = 0;

	return data_fwd_bme688_append(dev_bme, values, values_size, count);
}

uint8_t data_fwd_sensor_channel_count(void)
{
	return SENSOR_CHANNEL_COUNT;
}

const char *const *data_fwd_sensor_channel_names(void)
{
	return channel_names;
}

uint8_t data_fwd_sensor_type_id(void)
{
	return SENSOR_TYPE_VALUE;
}

uint16_t data_fwd_sensor_frequency(void)
{
	return FREQUENCY_HZ;
}
