/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup dmic DMIC control functions
 * @{
 * @ingroup keyword_spotting
 */

#ifndef __DMIC_H__
#define __DMIC_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define DMIC_SAMPLE_BYTES	(2)
#define DMIC_PCM_RATE		(16000)
#define SAMPLES_BLOCK_LENGTH_MS (10)

/**
 * @brief Initialize DMIC.
 *
 * @return Operation status result, 0 for success.
 */
int dmic_init(void);

/**
 * @brief Free the audio buffer acquired with  dmic_read.
 *
 * @param buffer Audio buffer.
 */
void free_dmic_buffer(void *buffer);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __DMIC_H__ */

/**
 * @}
 */
