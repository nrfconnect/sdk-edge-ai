/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef __BLE_NUS_H__
#define __BLE_NUS_H__

#ifdef __cplusplus
extern "C" {
#endif

int ble_nus_init(void);
void ble_nus_restart_connection(void);
int ble_nus_send_message(const char *message);

/** @brief Send one classification line, optionally tagged with @p prefix.
 *
 * Raw and published predictions share this formatting so that the two line
 * types stay field-for-field identical in a capture. Pass NULL for no prefix.
 */
int ble_nus_send_classification(const char *prefix, const char *class_name, int probability_pct,
				float accel_x_g, float accel_y_g, float accel_z_g);

#ifdef __cplusplus
}
#endif

#endif /* __BLE_NUS_H__ */
