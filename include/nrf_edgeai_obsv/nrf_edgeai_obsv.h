/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_obsv nRF Edge AI observability
 * @{
 *
 * @brief Mutex-protected context, portable state, metrics, and optional CBOR encoding.
 *
 */
#ifndef NRF_EDGEAI_OBSV_H
#define NRF_EDGEAI_OBSV_H

#include <stddef.h>
#include <stdint.h>

#include <zephyr/kernel.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_core.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Zephyr RTOS synchronization for one observability context.
 *
 * Embeds @ref nrf_edgeai_obsv_core_t with a mutex so @ref nrf_edgeai_obsv_update_probs,
 * @ref nrf_edgeai_obsv_encode / @ref nrf_edgeai_obsv_for_each_metric, and metric
 * registration can be serialized across threads.
 */
typedef struct nrf_edgeai_obsv_ctx {
	/** @brief Serializes observability API for this context. */
	struct k_mutex lock;
	/** @brief Portable observability state (shared with encode/snapshot paths). */
	nrf_edgeai_obsv_core_t state;
} nrf_edgeai_obsv_ctx_t;

/**
 * @brief Initialize an observability context.
 *
 * @param ctx   Context to initialize.
 * @param model Model metadata to embed in every encoded snapshot.
 * @return 0 on success, -EINVAL if any argument is NULL, if
 *         @p model->num_classes is 0, or if it exceeds
 *         @c CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES.
 */
int nrf_edgeai_obsv_init(nrf_edgeai_obsv_ctx_t *ctx, const nrf_edgeai_obsv_model_info_t *model);

/**
 * @brief Register a metric with the context.
 *
 * Calls @p metric->init(@p cfg) and appends @p metric to the context's list.
 * Returns -EALREADY without calling init if @p metric is already registered.
 *
 * @param ctx    Initialized context.
 * @param metric Metric descriptor to register.
 * @param cfg    Implementation-specific configuration passed to @p metric->init.
 *               May be NULL for metrics that accept default configuration.
 * @retval 0 Success; @p metric->init(@p cfg) was called and the metric was appended.
 * @retval -EINVAL @p ctx or @p metric is NULL, or @p metric->snapshot is NULL.
 * @retval -EALREADY @p metric was already registered (init not called again).
 */
int nrf_edgeai_obsv_register(nrf_edgeai_obsv_ctx_t *ctx, nrf_edgeai_obsv_metric_t *metric,
			     const void *cfg);

/**
 * @brief Deregister a previously registered metric.
 *
 * Removes @p metric from the context's list. No-op if @p metric is not found.
 *
 * @param ctx    Initialized context.
 * @param metric Metric descriptor to remove.
 * @return 0 on success, -EINVAL if @p ctx or @p metric is NULL.
 */
int nrf_edgeai_obsv_deregister(nrf_edgeai_obsv_ctx_t *ctx, nrf_edgeai_obsv_metric_t *metric);

/**
 * @brief Reset inference counters and all registered metrics.
 *
 * Calls each metric's @c clear callback to zero counters while preserving
 * any configuration set at registration time (e.g. custom bin edges).
 * Registered metrics remain attached; model metadata is preserved.
 *
 * @param ctx Initialized context.
 * @return 0 on success, -EINVAL if @p ctx is NULL.
 */
int nrf_edgeai_obsv_reset(nrf_edgeai_obsv_ctx_t *ctx);

/**
 * @brief Feed one inference result to all registered metrics.
 *
 * @param ctx   Initialized context.
 * @param probs Array of @c model.num_classes class probabilities.
 * @return 0 on success, -EINVAL if @p ctx or @p probs is NULL.
 */
int nrf_edgeai_obsv_update_probs(nrf_edgeai_obsv_ctx_t *ctx, const float *probs);

/**
 * @brief Feed one extracted-feature vector to all FEATURES-source metrics.
 *
 * Routes @p feats only to metrics registered with
 * @c NRF_EDGEAI_OBSV_SOURCE_FEATURES. Does not advance the inference counter;
 * an inference is counted when its output is fed via @ref nrf_edgeai_obsv_update_probs.
 *
 * @param ctx   Initialized context.
 * @param feats Array of @p n feature values.
 * @param n     Number of entries in @p feats.
 * @return 0 on success, -EINVAL if @p ctx or @p feats is NULL.
 */
int nrf_edgeai_obsv_update_features(nrf_edgeai_obsv_ctx_t *ctx, const float *feats, uint16_t n);

/**
 * @brief Iterate over all registered metrics under the context lock.
 *
 * @param ctx  Initialized context.
 * @param cb   Callback invoked once per metric snapshot. Return false to stop early.
 * @param user Opaque pointer forwarded to @p cb.
 * @return 0 on success, -EINVAL if @p ctx or @p cb is NULL.
 */
int nrf_edgeai_obsv_for_each_metric(nrf_edgeai_obsv_ctx_t *ctx, nrf_edgeai_obsv_metric_cb_t cb,
				    void *user);

/**
 * @brief Encode all metrics as a CBOR blob under the context lock.
 *
 * Wire format is defined in @c obsv.cddl. Size @p buf for the worst-case
 * encoded payload (see @c nrf_edgeai_obsv_encode.c for the formula).
 *
 * Requires @c CONFIG_NRF_EDGEAI_OBSV_ENCODE.
 *
 * @param ctx     Initialized context.
 * @param buf     Output buffer.
 * @param max_len Size of @p buf in bytes.
 * @return Encoded byte count on success, 0 on error or buffer overflow.
 */
size_t nrf_edgeai_obsv_encode(nrf_edgeai_obsv_ctx_t *ctx, uint8_t *buf, size_t max_len);

/**
 * @brief Encode multiple contexts as a CBOR list (obsv-list format).
 *
 * Produces the wire format @c obsv-list = [+ obsv-payload] defined in
 * @c obsv.cddl: a CBOR array with one @c obsv-payload element per context,
 * each encoded under its own context lock. Suitable for any transport
 * (Memfault CDR, UART, BLE, etc.).
 *
 * Requires @c CONFIG_NRF_EDGEAI_OBSV_ENCODE.
 *
 * @param ctxs    Array of @p n initialized context pointers.
 * @param n       Number of contexts. Must be 1–23 (CBOR tiny integer limit).
 * @param buf     Output buffer.
 * @param max_len Size of @p buf in bytes.
 * @return Encoded byte count on success, 0 on error or buffer overflow.
 */
size_t nrf_edgeai_obsv_encode_list(nrf_edgeai_obsv_ctx_t *const *ctxs, uint8_t n,
				   uint8_t *buf, size_t max_len);

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_H */

/**
 * @}
 */
