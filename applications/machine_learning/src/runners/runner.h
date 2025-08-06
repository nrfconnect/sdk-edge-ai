/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef _RUNNER_H_
#define _RUNNER_H_

#include <stdbool.h>
#include <stddef.h>

/**
 * @typedef runner_continue_callback
 * @brief Callback executed by runner after inference completed.
 *
 * @param err Zero if inference was successful or negative error code.
 *
 * @retval true If upper module is not in error state.
 */
typedef bool (*runner_continue_callback)(int err);

/** Initialize runner.
 *
 * @param[in] cb Callback used to indicate inference completion.
 *
 * @retval 0 If the operation was successful.
 *           Otherwise, a negative error code.
 */
int runner_init(const runner_continue_callback cb);

/** Start a prediction.
 *
 * If there is not enough data in the internal buffer, the inference start is delayed until the
 * missing data is added.
 * Not all runners implement this function, they will run inference whenever enough data was added.
 *
 * @retval 0       If the operation was successful.
 * @retval -ENOSYS If not implemented.
 *                 Otherwise, a negative error code.
 */
int runner_start_prediction(void);

/** Stop a prediction.
 *
 * This does not interrupt already running inference. Clear internal buffer if possible.
 * Not all runners implement this function, as they do not expose buffer.
 *
 * @retval 0       If the operation was successful.
 * @retval -ENOSYS If not implemented.
 *                 Otherwise, a negative error code.
 */
int runner_stop_prediction(void);

/** Add input data for runner.
 *
 * Size of the added data must be divisible by input frame size.
 *
 * @param[in] data       Pointer to the buffer with input data.
 * @param[in] data_size  Size of the data (number of floating-point values).
 *
 * @retval 0 If the operation was successful.
 *           Otherwise, a negative error code.
 */
int runner_add_data(const float *data, size_t data_size);

#endif /* _RUNNER_H_ */
