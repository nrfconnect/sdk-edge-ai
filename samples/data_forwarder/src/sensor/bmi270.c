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

#define BMI270_NODE	     DT_NODELABEL(bmi270)
#define BMI270_CHANNEL_COUNT 6
#define SENSOR_TYPE_VALUE    1
#define FREQUENCY_HZ	     100

BUILD_ASSERT(FREQUENCY_HZ > 0, "Sampling frequency must be non-zero");

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
#define SENSOR_CHANNEL_COUNT (BMI270_CHANNEL_COUNT + BME688_CHANNEL_COUNT)
#else
#define SENSOR_CHANNEL_COUNT BMI270_CHANNEL_COUNT
#endif

static const char *const channel_names[SENSOR_CHANNEL_COUNT] = {
	"ax", "ay", "az", "gx", "gy", "gz",
#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
	BME688_CHANNEL_NAMES
#endif
};

static const struct device *dev_bmi = DEVICE_DT_GET(BMI270_NODE);

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
DATA_FWD_BME688_RATE_ASSERT(FREQUENCY_HZ);

static const struct device *dev_bme = DEVICE_DT_GET(BME688_NODE);
static struct bme688_work_wrapper bme688_work_wrapper = {
	.work = Z_WORK_INITIALIZER(bme688_work_handler),
	.dev = DEVICE_DT_GET(BME688_NODE),
};
#endif

static K_SEM_DEFINE(fetch_sem, 0, 1);
static void fetch_timer_handler(struct k_timer *timer);
static K_TIMER_DEFINE(fetch_timer, fetch_timer_handler, NULL);

static void fetch_timer_handler(struct k_timer *timer)
{
	ARG_UNUSED(timer);
	k_sem_give(&fetch_sem);
}

static bool bmi_set_attr(enum sensor_channel chan, enum sensor_attribute attr,
			 const struct sensor_value *val)
{
	int err = sensor_attr_set(dev_bmi, chan, attr, val);

	if (err) {
		LOG_ERR("Sensor attribute set failed (err %d)", err);
		return false;
	}

	return true;
}

int data_fwd_sensor_init(void)
{
	struct sensor_value full_scale, sampling_freq, oversampling;

	if (!device_is_ready(dev_bmi)) {
		return -ENODEV;
	}

#if IS_ENABLED(CONFIG_DATA_FWD_EXTRA_SENSOR_BME688)
	if (!device_is_ready(dev_bme)) {
		return -ENODEV;
	}
#endif

	/* Setting scale in G to match the sensor scale */
	full_scale.val1 = 2; /* G */
	full_scale.val2 = 0;
	sampling_freq.val1 = FREQUENCY_HZ; /* Hz. Performance mode */
	sampling_freq.val2 = 0;
	oversampling.val1 = 1; /* Normal mode */
	oversampling.val2 = 0;

	bool ok = true;

	ok &= bmi_set_attr(SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_FULL_SCALE, &full_scale);
	ok &= bmi_set_attr(SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_OVERSAMPLING, &oversampling);
	/* Set sampling frequency last as this also sets the appropriate power mode. If already
	 * sampling, change to 0.0Hz before changing other attributes
	 */
	ok &= bmi_set_attr(SENSOR_CHAN_ACCEL_XYZ, SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

	/* Setting scale in degrees/s to match the sensor scale */
	full_scale.val1 = 500; /* dps */
	full_scale.val2 = 0;
	sampling_freq.val1 = FREQUENCY_HZ; /* Hz. Performance mode */
	sampling_freq.val2 = 0;
	oversampling.val1 = 1; /* Normal mode */
	oversampling.val2 = 0;

	ok &= bmi_set_attr(SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_FULL_SCALE, &full_scale);
	ok &= bmi_set_attr(SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_OVERSAMPLING, &oversampling);
	/* Set sampling frequency last as this also sets the appropriate power mode. If already
	 * sampling, change sampling frequency to 0.0Hz before changing other attributes
	 */
	ok &= bmi_set_attr(SENSOR_CHAN_GYRO_XYZ, SENSOR_ATTR_SAMPLING_FREQUENCY, &sampling_freq);

	if (!ok) {
		return -ENOTSUP;
	}

	const uint32_t period_ns = Z_HZ_ns / FREQUENCY_HZ;

	k_timer_start(&fetch_timer, K_NO_WAIT, K_NSEC(period_ns));

	return 0;
}

int data_fwd_sensor_fetch(proto_value_t *values, const size_t values_size, size_t *count)
{
	int err;

	struct sensor_value accel[3] = {0};
	struct sensor_value gyro[3] = {0};

	if ((values == NULL) || (count == NULL) || (values_size < SENSOR_CHANNEL_COUNT)) {
		return -EINVAL;
	}

	k_sem_take(&fetch_sem, K_FOREVER);

	err = sensor_sample_fetch(dev_bmi);
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

	err = sensor_channel_get(dev_bmi, SENSOR_CHAN_ACCEL_XYZ, accel);
	if (err) {
		return err;
	}

	err = sensor_channel_get(dev_bmi, SENSOR_CHAN_GYRO_XYZ, gyro);
	if (err) {
		return err;
	}

	for (size_t i = 0; i < 3; i++) {
		values[i] = data_fwd_sensor_value_to_proto_value(&accel[i]);
		values[i + 3] = data_fwd_sensor_value_to_proto_value(&gyro[i]);
	}

	*count = BMI270_CHANNEL_COUNT;

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
