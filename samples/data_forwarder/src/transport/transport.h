/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef DATA_FORWARDER_TRANSPORT_H_
#define DATA_FORWARDER_TRANSPORT_H_

#include "../protocol/protocol.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize the selected transport and fill @p out_transport.
 *
 * The concrete backend is chosen at build time via @c CONFIG_DATA_FWD_TRANSPORT_*.
 *
 * @param[out] out_transport Populated with send callback and transport settings.
 *
 * @retval 0 Success.
 * @retval -EINVAL @p out_transport is @c NULL.
 * @retval -errno Negative error code on failure.
 */
int transport_init(struct proto_transport *out_transport);

/**
 * @brief Return whether the transport link is ready to carry data.
 *
 * For BLE NUS, this is true while a central is connected.
 * For UART, this is always true.
 */
bool transport_is_connected(void);

/**
 * @brief Block until the transport link becomes ready.
 *
 * For BLE NUS, this waits for a central to connect.
 * For UART, this returns immediately.
 */
void transport_wait_connected(void);

#ifdef __cplusplus
}
#endif

#endif /* DATA_FORWARDER_TRANSPORT_H_ */
