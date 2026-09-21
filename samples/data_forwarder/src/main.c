/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>

#include "protocol/protocol.h"
#include "sensor/data_fwd_sensor.h"
#include "transport/transport.h"

LOG_MODULE_REGISTER(data_forwarder);

/**
 * @brief Stop sensor sampling and log a warning if stop fails.
 */
static void sensor_stop(void)
{
	int err = data_fwd_sensor_stop();

	if (err) {
		LOG_WRN("Sensor stop failed (err %d)", err);
	}
}

/**
 * @brief Start sensors and a protocol session, retrying on failure.
 *
 * Retries while the transport link remains up. Returns when both starts
 * succeed or the link drops.
 *
 * @param session Session configuration passed to @ref proto_start_session().
 */
static void session_start_retry(const struct proto_session_config *session)
{
	int err;

	while (transport_is_connected()) {
		err = data_fwd_sensor_start();
		if (err) {
			LOG_WRN("Sensor start failed (err %d), retrying in %d ms", err,
				CONFIG_DATA_FWD_START_RETRY_MS);
			k_sleep(K_MSEC(CONFIG_DATA_FWD_START_RETRY_MS));
			continue;
		}

		err = proto_start_session(session);
		if (err) {
			LOG_WRN("Failed to start session (err %d), retrying in %d ms", err,
				CONFIG_DATA_FWD_START_RETRY_MS);
			sensor_stop();
			k_sleep(K_MSEC(CONFIG_DATA_FWD_START_RETRY_MS));
			continue;
		}

		LOG_INF("Sampling session started (sid %u)", proto_get_session_id());
		break;
	}
}

/**
 * @brief Fetch sensor samples and send them until the transport disconnects.
 */
static void stream_samples(void)
{
	int err;

	while (transport_is_connected()) {
		proto_value_t values[CONFIG_DATA_FWD_PROTO_MAX_CHANNELS];
		size_t count;

		err = data_fwd_sensor_fetch(values, ARRAY_SIZE(values), &count);
		if (err) {
			LOG_WRN("Sample fetch failed (err %d)", err);
			continue;
		}

		err = proto_send_samples(values, count);
		if (err) {
			LOG_WRN("Sample send failed (err %d)", err);
		}
	}
}

/**
 * @brief Stop the protocol session and sensor sampling after a connection ends.
 */
static void session_stop(void)
{
	proto_stop_session();
	sensor_stop();
	LOG_INF("Connection terminated");
}

int main(void)
{
	int err;

	struct proto_transport transport;
	const struct proto_session_config session = {
		.rate_hz = data_fwd_sensor_frequency(),
		.channels = data_fwd_sensor_channel_count(),
		.sensor_type = data_fwd_sensor_type_id(),
		.channel_names = data_fwd_sensor_channel_names(),
#if defined(CONFIG_BT_DEVICE_NAME)
		.device_name = CONFIG_BT_DEVICE_NAME,
#else
		.device_name = "nRF DataFwd",
#endif
	};

	err = transport_init(&transport);
	if (err) {
		LOG_ERR("Transport init failed (err %d)", err);
		return err;
	}

	err = proto_init(&transport);
	if (err) {
		LOG_ERR("Protocol init failed (err %d)", err);
		return err;
	}

	err = data_fwd_sensor_init();
	if (err) {
		LOG_ERR("Sensor init failed (err %d)", err);
		return err;
	}

	LOG_INF("Data forwarder started");

	while (1) {
		transport_wait_connected();
		session_start_retry(&session);
		stream_samples();
		session_stop();
	}

	return 0;
}
