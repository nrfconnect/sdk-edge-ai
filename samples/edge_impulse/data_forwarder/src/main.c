/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <stdio.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/logging/log.h>

#define LOG_MODULE_NAME data_forwarder
LOG_MODULE_REGISTER(LOG_MODULE_NAME);

#define SAMPLE_PERIOD_MS	100
#define UART_BUF_SIZE		64
#define NUM_OF_SENSOR_CHANNELS	ARRAY_SIZE(sensor_channels)

static void sensor_timer_handler(struct k_timer *timer_id);

K_TIMER_DEFINE(sensor_timer, sensor_timer_handler, NULL);
K_SEM_DEFINE(sensor_sem, 0, 1);

const static enum sensor_channel sensor_channels[] = {
	SENSOR_CHAN_ACCEL_X,
	SENSOR_CHAN_ACCEL_Y,
	SENSOR_CHAN_ACCEL_Z
};

static const struct device *sensor_dev = DEVICE_DT_GET(DT_NODELABEL(sensor_sim));
static const struct device *uart_dev = DEVICE_DT_GET(DT_CHOSEN(ncs_ei_data_forwarder_uart));
static atomic_t uart_busy;
static atomic_t processing_sensor_data;


static void uart_cb(const struct device *dev, struct uart_event *evt,
		    void *user_data)
{
	if (evt->type == UART_TX_DONE) {
		(void)atomic_set(&uart_busy, false);
	}
}

static int init(void)
{
	atomic_set(&uart_busy, false);
	atomic_set(&processing_sensor_data, false);

	if (!device_is_ready(sensor_dev)) {
		LOG_ERR("Sensor device not ready");
		return -ENODEV;
	}

	if (!device_is_ready(uart_dev)) {
		LOG_ERR("UART device not ready");
		return -ENODEV;
	}

	int err = uart_callback_set(uart_dev, uart_cb, NULL);

	if (err) {
		LOG_ERR("Cannot set UART callback (err %d)", err);
	}

	return err;
}

static int provide_sensor_data(void)
{
	struct sensor_value data[NUM_OF_SENSOR_CHANNELS];
	int err = 0;

	/* Sample simulated sensor. */
	for (size_t i = 0; (!err) && (i < NUM_OF_SENSOR_CHANNELS); i++) {
		err = sensor_sample_fetch_chan(sensor_dev, sensor_channels[i]);
		if (!err) {
			err = sensor_channel_get(sensor_dev, sensor_channels[i],
						 &data[i]);
		}
	}

	if (err) {
		LOG_ERR("Sensor sampling error (err %d)", err);
		return err;
	}

	/* Send data over UART. */
	static uint8_t buf[UART_BUF_SIZE];

	/* Protect UART and buffer access. */
	if (!atomic_cas(&uart_busy, false, true)) {
		LOG_ERR("UART not ready - please use lower sampling frequency");
		return -EBUSY;
	}

	BUILD_ASSERT(3 == NUM_OF_SENSOR_CHANNELS,
		     "Output format of snprintf assumes 3 sensor channels");

	/* Prepare format expected by edge-impulse-data-forwarder. */
	int res = snprintf((char *)buf, sizeof(buf), "%.2f,%.2f,%.2f\r\n",
			   sensor_value_to_double(&data[0]),
			   sensor_value_to_double(&data[1]),
			   sensor_value_to_double(&data[2]));

	if (res < 0) {
		LOG_ERR("snprintf returned error (err %d)", res);
		return res;
	} else if (res >= sizeof(buf)) {
		LOG_ERR("UART_BUF_SIZE is too small to store the data - %d bytes are required",
			res);
		return -ENOMEM;
	}

	err = uart_tx(uart_dev, buf, res, SYS_FOREVER_MS);

	if (err) {
		LOG_ERR("Cannot send data over UART (err %d)", err);
		atomic_set(&uart_busy, false);
	} else {
		LOG_DBG("Sent data: %s", buf);
	}

	return err;
}

static void sensor_timer_handler(struct k_timer *timer_id)
{
	ARG_UNUSED(timer_id);

	if (atomic_get(&processing_sensor_data)) {
		LOG_ERR("Previous sensor read still pending - missed sample!");
	}

	k_sem_give(&sensor_sem);
}

int main(void)
{
	int err = init();
	if (err) {
		return err;
	}

	LOG_INF("Initialization done. Starting data forwarding...");

	/* Start the timer to provide sensor data every SAMPLE_PERIOD_MS milliseconds. */
	k_timer_start(&sensor_timer, K_MSEC(SAMPLE_PERIOD_MS), K_MSEC(SAMPLE_PERIOD_MS));

	while (1) {
		k_sem_take(&sensor_sem, K_FOREVER);
		atomic_set(&processing_sensor_data, true);
		provide_sensor_data();
		atomic_set(&processing_sensor_data, false);
	}

	return 0;
}
