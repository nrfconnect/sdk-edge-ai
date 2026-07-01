/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>

#include "data_fwd_sensor.h"
#include "bme688_common.h"

LOG_MODULE_REGISTER(sensor);

#define ADXL367_NODE	      DT_NODELABEL(adxl367)
#define ADXL367_CHANNEL_COUNT 3
#define SENSOR_TYPE_VALUE     2
#define FREQUENCY_HZ	      50

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
#define SENSOR_CHANNEL_COUNT (ADXL367_CHANNEL_COUNT + BME688_CHANNEL_COUNT)
#else
#define SENSOR_CHANNEL_COUNT ADXL367_CHANNEL_COUNT
#endif

static const char *const channel_names[SENSOR_CHANNEL_COUNT] = {
	"ax", "ay", "az",
#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
	BME688_CHANNEL_NAMES
#endif
};

static const struct device *dev_adxl = DEVICE_DT_GET(ADXL367_NODE);

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
DATA_FWD_BME688_RATE_ASSERT(FREQUENCY_HZ);

static const struct device *dev_bme = DEVICE_DT_GET(BME688_NODE);
static struct bme688_work_wrapper bme688_work_wrapper = {
	.work = Z_WORK_INITIALIZER(bme688_work_handler),
	.dev = DEVICE_DT_GET(BME688_NODE),
};
#endif

BUILD_ASSERT(DT_PROP(ADXL367_NODE, odr) == 2, "Adjust FREQUENCY_HZ macro after changing dts");

int data_fwd_sensor_init(void)
{
	if (!device_is_ready(dev_adxl)) {
		return -ENODEV;
	}

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
	if (!device_is_ready(dev_bme)) {
		return -ENODEV;
	}
#endif

	return 0;
}

int data_fwd_sensor_fetch(proto_value_t *values, const size_t values_size, size_t *count)
{
	int err;

	struct sensor_value accel[ADXL367_CHANNEL_COUNT] = {0};

	if ((values == NULL) || (count == NULL) || (values_size < SENSOR_CHANNEL_COUNT)) {
		return -EINVAL;
	}

	/* Use ADXL367 driver implementation for timing sampling interval. It blocks until values
	 * are ready to fetch.
	 */
	err = sensor_sample_fetch(dev_adxl);
	if (err) {
		return err;
	}

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
	static size_t bme_counter;
	const bool fetch_bme = data_fwd_bme688_should_fetch(&bme_counter, FREQUENCY_HZ);

	if (fetch_bme) {
		k_work_submit(&bme688_work_wrapper.work);
	}
#endif

	err = sensor_channel_get(dev_adxl, SENSOR_CHAN_ACCEL_XYZ, accel);
	if (err) {
		return err;
	}

	for (size_t i = 0; i < ADXL367_CHANNEL_COUNT; i++) {
		values[i] = data_fwd_sensor_value_to_proto_value(&accel[i]);
	}

	*count = ADXL367_CHANNEL_COUNT;

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
	/* BME688 is fetched asynchronously; appended values may be from the previous cycle. */
	err = data_fwd_bme688_append(dev_bme, values + *count, values_size - *count, count);
	if (err) {
		return err;
	}
#endif

	return 0;
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
