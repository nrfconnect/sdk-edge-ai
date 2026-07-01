/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef DATA_FORWARDER_SENSOR_H_
#define DATA_FORWARDER_SENSOR_H_

/**
 * @file
 * @defgroup sample_data_fwd Data Forwarder Sample
 * @defgroup sample_data_fwd_sensor Sensor wrapper
 * @ingroup sample_data_fwd
 * @{
 */

#include <stddef.h>
#include <stdint.h>

#include <zephyr/drivers/sensor.h>
#include <zephyr/sys/util.h>

#include "../protocol/protocol_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the selected sensor driver.
 *
 * The concrete driver is chosen at build time via @c CONFIG_DATA_FWD_SENSOR_*.
 *
 * @retval 0 Success.
 * @retval -ENODEV Sensor device is not ready.
 * @retval -ENOTSUP Sensor configuration failed (driver-specific).
 */
int data_fwd_sensor_init(void);

/**
 * @brief Read one sample from the sensor into @p values.
 *
 * @note This function is blocking until sensor values are available.
 *
 * @param values       Output buffer for channel values.
 * @param values_size  Capacity of @p values in elements.
 * @param count        On success, set to the number of values written.
 *
 * @retval 0 Success.
 * @retval -EINVAL @p values, @p count, or @p values_size is invalid.
 * @retval -errno Negative error code on fetch failure.
 */
int data_fwd_sensor_fetch(proto_value_t *values, const size_t values_size, size_t *count);

/**
 * @brief Get the number of data channels in each sample.
 *
 * @return Channel count for the selected sensor.
 */
uint8_t data_fwd_sensor_channel_count(void);

/**
 * @brief Get short names for each data channel.
 *
 * The returned array has @ref data_fwd_sensor_channel_count() entries and remains valid
 * for the lifetime of the application.
 *
 * @return Pointer to an array of null-terminated channel name strings.
 */
const char *const *data_fwd_sensor_channel_names(void);

/**
 * @brief Get the application-defined sensor type identifier.
 *
 * @return Sensor type ID.
 */
uint8_t data_fwd_sensor_type_id(void);

/**
 * @brief Get the nominal sampling frequency in Hz.
 *
 * @return Sampling rate in Hz for the selected sensor.
 */
uint16_t data_fwd_sensor_frequency(void);

/**
 * @brief Convert a Zephyr sensor value to a protocol sample value.
 *
 * When @c CONFIG_DATA_FWD_PROTO_INT32_VALUES is enabled, the result is clamped to
 * @c INT32_MIN..@c INT32_MAX micro-units. Otherwise a float is returned.
 *
 * @param val Zephyr sensor value to convert.
 *
 * @return Value encoded for the active protocol mode.
 */
static inline proto_value_t data_fwd_sensor_value_to_proto_value(const struct sensor_value *val)
{
#if IS_ENABLED(CONFIG_DATA_FWD_PROTO_INT32_VALUES)
	return (proto_value_t)CLAMP(sensor_value_to_micro(val), INT32_MIN, INT32_MAX);
#else
	return sensor_value_to_float(val);
#endif
}

#ifdef __cplusplus
}
#endif

/**
 * @}
 */

#endif /* DATA_FORWARDER_SENSOR_H_ */
