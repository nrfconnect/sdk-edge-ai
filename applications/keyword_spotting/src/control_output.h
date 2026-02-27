/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup control_output Printing control output functions
 * @{
 * @ingroup keyword_spotting
 */

#ifndef __CONTROL_OUTPUT_H__
#define __CONTROL_OUTPUT_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Control message type.
 */
enum control_message_type {
	CONTROL_MESSAGE_WAITING_WW,
	CONTROL_MESSAGE_WW_DETECTED,
};

/**
 * @brief Control message.
 */
struct control_message {
	enum control_message_type type;
};

/**
 * @brief Initialize control output backend.
 *
 * @return Operation status, 0 for success.
 */
int control_output_init(void);

/**
 * @brief Initialize LEDs.
 *
 * @param message Control message to be printed.
 */
void print_control_output(const struct control_message message);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __CONTROL_OUTPUT_H__ */

/**
 * @}
 */
