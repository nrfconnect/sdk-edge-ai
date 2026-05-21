/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_obsv_encode_cbor CBOR encoding (no mutex)
 * @{
 * @ingroup nrf_edgeai_obsv_encode
 *
 * @brief Encodes an observability snapshot to CBOR without acquiring any mutex.
 *
 */
#ifndef NRF_EDGEAI_OBSV_ENCODE_H
#define NRF_EDGEAI_OBSV_ENCODE_H

#include <stddef.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Encode observability state and metrics as CBOR (no mutex).
 *
 * Wire format is defined in obsv.cddl. On Zephyr, prefer @ref nrf_edgeai_obsv_encode
 * on @c nrf_edgeai_obsv_ctx_t* so the context mutex serializes with inference.
 *
 * @param state Initialized portable state with registered metrics.
 * @param buf Output buffer.
 * @param max_len Size of @p buf.
 * @return Encoded byte count on success, 0 on encoding failure or overflow.
 *
 * @note Every registered metric is emitted. The required buffer size depends
 * on the registered metrics and their Kconfig dimensions; custom or larger
 * snapshots need a correspondingly larger @p buf and @p max_len.
 */
size_t nrf_edgeai_obsv_encode_cbor(nrf_edgeai_obsv_core_t *state, uint8_t *buf, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_ENCODE_H */

/**
 * @}
 */
