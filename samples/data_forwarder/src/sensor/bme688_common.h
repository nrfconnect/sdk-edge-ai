/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef DATA_FORWARDER_BME688_COMMON_H_
#define DATA_FORWARDER_BME688_COMMON_H_

#include <zephyr/drivers/sensor.h>

#include "data_fwd_sensor.h"

#define BME688_NODE	     DT_NODELABEL(bme688)
#define BME688_CHANNEL_COUNT 3
#define BME688_FREQUENCY_HZ  1
#define BME688_CHANNEL_NAMES "temp", "hum", "pres"

#define DATA_FWD_BME688_RATE_ASSERT(main_hz)                                                       \
	BUILD_ASSERT((main_hz) % BME688_FREQUENCY_HZ == 0,                                         \
		     "Main sensor frequency must be a multiple of BME688 frequency")

struct bme688_work_wrapper {
	struct k_work work;
	const struct device *dev;
};

static inline bool data_fwd_bme688_should_fetch(size_t *counter, uint16_t main_hz)
{
	const bool fetch = (*counter == 0);

	*counter = (*counter + 1) % (main_hz / BME688_FREQUENCY_HZ);

	return fetch;
}

static inline int data_fwd_bme688_fetch(const struct device *dev)
{
	if (dev == NULL) {
		return -EINVAL;
	}

	return sensor_sample_fetch(dev);
}

static inline int data_fwd_bme688_append(const struct device *dev, proto_value_t *values,
					 const size_t values_size, size_t *count)
{
	int err;

	struct sensor_value temp = {0};
	struct sensor_value humidity = {0};
	struct sensor_value pressure = {0};

	if ((dev == NULL) || (values == NULL) || (count == NULL) ||
	    (values_size < BME688_CHANNEL_COUNT)) {
		return -EINVAL;
	}

	err = sensor_channel_get(dev, SENSOR_CHAN_AMBIENT_TEMP, &temp);
	if (err) {
		return err;
	}

	err = sensor_channel_get(dev, SENSOR_CHAN_HUMIDITY, &humidity);
	if (err) {
		return err;
	}

	err = sensor_channel_get(dev, SENSOR_CHAN_PRESS, &pressure);
	if (err) {
		return err;
	}

	values[0] = data_fwd_sensor_value_to_proto_value(&temp);
	values[1] = data_fwd_sensor_value_to_proto_value(&humidity);
	values[2] = data_fwd_sensor_value_to_proto_value(&pressure);

	*count += BME688_CHANNEL_COUNT;

	return 0;
}

static inline void bme688_work_handler(struct k_work *work)
{
	struct bme688_work_wrapper *wrapper = CONTAINER_OF(work, struct bme688_work_wrapper, work);

	data_fwd_bme688_fetch(wrapper->dev);
}

#endif /* DATA_FORWARDER_BME688_COMMON_H_ */
