/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup keyword_spotting Keyword Spotting application
 */

/**
 * @defgroup wakeword Wakeword model functions
 * @{
 * @ingroup keyword_spotting
 */

#ifndef __WAKEWORD_H__
#define __WAKEWORD_H__

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Initialize Wakeword model.
 *
 * @return Operation status, 0 for success.
 */
int ww_init(void);

/**
 * @brief Process audio data by Wakeword model.
 *
 * @param audio_buffer Buffer of audio samples from dmic_read. Function takes ownership of pointer.
 * @param num_samples Number of audio samples.
 * @param[out] ww_detected Result of wakeword detection. Valid if operation completed successfully.

 * @retval 0 Operation successful.
 * @retval -EPERM Operation failed on nRF Edge AI Lib level.
 * @retval -EBUSY Model needs more data.
 */
int ww_process(uint8_t *const audio_buffer, const uint16_t num_samples, bool *const ww_detected);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __WAKEWORD_H__ */

/**
 * @}
 */
