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
 * Encode during collect is serialized against @ref nrf_edgeai_obsv_update_probs using
 * @p ctx->lock so snapshots cannot race inference on this context.
 *
 * @param ctx Initialized observability context (@c nrf_edgeai_obsv_init).
 * @return 0 on success, negative errno on failure.
 */
int nrf_edgeai_obsv_memfault_init(struct nrf_edgeai_obsv_ctx *ctx);

/**
 * @brief Encodes observability metrics as CBOR and stages them for Memfault.
 *
 * The CBOR blob is stored internally and handed out by the registered CDR source
 * on the next transport drain cycle. Every registered context is reset in the
 * same critical section as its encode, so each payload covers exactly the
 * interval since the previous successful collect.
 *
 * A staged payload is never overwritten. While the previous payload has not been
 * drained, the function returns @c -EBUSY and leaves the contexts untouched, so
 * their data keeps accumulating into the next payload. With
 * @c CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT, a refused collect is retried
 * as soon as the payload is drained; otherwise call this function again later.
 *
 * @note This function acquires the locks (@c ctx->lock) of all registered
 * observability contexts while encoding and resetting, thus could lead to stall
 * of inference pipeline when called from low priority threads.
 * @note Allocates a @c NRF_EDGEAI_OBSV_ENCODE_LIST_BUFSZ-byte buffer
 * on the calling thread's stack. When invoked from the system workqueue
 * (e.g. via @c CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT), increase
 * @c CONFIG_SYSTEM_WORKQUEUE_STACK_SIZE accordingly.
 *
 * @retval 0        Success; payload staged and ready for the next drain cycle.
 * @retval -EINVAL  Not initialized (no context registered).
 * @retval -EBUSY   The previous payload has not been drained yet, or another
 *                  collect is in progress. Nothing was encoded or reset.
 * @retval -ENODATA CBOR encoding failed. Contexts were not reset.
 */
int nrf_edgeai_obsv_memfault_collect(void);

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_MEMFAULT_H */

/**
 * @}
 */
