/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_obsv_memfault Memfault CDR transport
 * @{
 * @ingroup nrf_edgeai_obsv
 *
 * @brief Stages CBOR-encoded observability snapshots as Memfault Custom Data Recordings.
 *
 */
#ifndef NRF_EDGEAI_OBSV_MEMFAULT_H
#define NRF_EDGEAI_OBSV_MEMFAULT_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nrf_edgeai_obsv_ctx;

/**
 * @brief Binds the Memfault CDR transport to a Zephyr observability context.
 *
 * Registers a Memfault Custom Data Recording source that serves the
 * CBOR-encoded observability payload when the packetizer drains data.
 * Must be called once before nrf_edgeai_obsv_memfault_collect().
 *
 * Encode during collect is serialized against @ref nrf_edgeai_obsv_update using
 * @p ctx->lock so snapshots cannot race inference on this context.
 *
 * @param ctx Initialized observability context (@c nrf_edgeai_obsv_init).
 * @return 0 on success, negative errno on failure.
 */
int nrf_edgeai_obsv_memfault_init(struct nrf_edgeai_obsv_ctx *ctx);

/**
 * @brief Encodes observability metrics as CBOR and stages them for Memfault.
 *
 * Overwrites any previously staged payload. The CBOR blob is stored internally
 * and handed out by the registered CDR source on the next transport drain cycle.
 *
 * @note This function acquires each registered observability context's
 * lock (@c ctx->lock) while encoding.
 * @note Allocates a @c NRF_EDGEAI_OBSV_ENCODE_LIST_BUFSZ-byte buffer
 * on the calling thread's stack. When invoked from the system workqueue
 * (e.g. via @c CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT), increase
 * @c CONFIG_SYSTEM_WORKQUEUE_STACK_SIZE accordingly.
 *
 * @retval 0        Success; payload staged and ready for the next drain cycle.
 * @retval -EINVAL  Not initialized (no context registered).
 * @retval -ENODATA CBOR encoding failed.
 */
int nrf_edgeai_obsv_memfault_collect(void);

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_MEMFAULT_H */

/**
 * @}
 */
