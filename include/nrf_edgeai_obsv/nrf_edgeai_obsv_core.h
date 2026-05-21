/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_obsv_core Portable observability state
 * @{
 * @ingroup nrf_edgeai_obsv
 *
 * @brief Mutex-free observability core usable without Zephyr dependencies.
 *
 */
#ifndef NRF_EDGEAI_OBSV_CORE_H
#define NRF_EDGEAI_OBSV_CORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Static model metadata included in observability snapshots.
 */
typedef struct nrf_edgeai_obsv_model_info_s {
	/** @brief Application-defined model identifier. */
	uint16_t model_id;
	/** @brief Number of model output classes. */
	uint16_t num_classes;
	/** @brief Model version identifier. */
	uint32_t version;
} nrf_edgeai_obsv_model_info_t;

/**
 * @brief Portable observability core (no RTOS).
 *
 * Holds model metadata, registered metric list, and accumulated counters.
 * Zephyr applications normally embed this in @c nrf_edgeai_obsv_ctx_t and call
 * @c nrf_edgeai_obsv_* so access is mutex-serialized.
 */
typedef struct {
	/** @brief Model metadata copied during initialization. */
	nrf_edgeai_obsv_model_info_t model;
	/** @brief Head of registered metrics singly linked list. */
	nrf_edgeai_obsv_metric_t *p_metrics_list;
	/** @brief Total number of processed inferences. Wraps at UINT32_MAX. */
	uint32_t num_inferences;
	/** @brief Total number of registered metrics. */
	uint8_t num_metrics;
} nrf_edgeai_obsv_core_t;

/**
 * @brief Callback invoked once per registered metric by
 *        nrf_edgeai_obsv_core_for_each_metric().
 */
typedef bool (*nrf_edgeai_obsv_metric_cb_t)(const nrf_edgeai_obsv_metric_snapshot_t *snap,
					    void *user);

/**
 * @brief Initialize portable observability core.
 *
 * Zeroes @p p_ctx and copies @p p_model into it. The mutex-less counterpart
 * of @ref nrf_edgeai_obsv_init, used by the Zephyr wrapper.
 *
 * @param p_ctx   Core to initialize.
 * @param p_model Model metadata to embed in every snapshot.
 * @return 0 on success, -EINVAL if either argument is NULL or if
 *         @p p_model->num_classes exceeds @c CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES.
 */
int nrf_edgeai_obsv_core_init(nrf_edgeai_obsv_core_t *p_ctx,
			      const nrf_edgeai_obsv_model_info_t *p_model);

/**
 * @brief Reset inference counter and clear all registered metrics.
 *
 * Calls each metric's @c clear callback to zero counters while preserving
 * configuration set at registration time. Metrics without a @c clear callback
 * are skipped (no-op).
 * Registered metrics remain attached; model metadata is preserved.
 *
 * @param p_ctx Initialized core.
 * @return 0 on success, -EINVAL if @p p_ctx is NULL.
 */
int nrf_edgeai_obsv_core_reset(nrf_edgeai_obsv_core_t *p_ctx);

/**
 * @brief Register a metric with the core.
 *
 * Calls @p p_metric->init(@p p_cfg) and appends @p p_metric to the tail of
 * the metric list. Returns -EALREADY without calling init if @p p_metric is
 * already in the list.
 *
 * @param p_ctx    Initialized core.
 * @param p_metric Metric descriptor to register.
 * @param p_cfg    Implementation-specific configuration forwarded to
 *                 @p p_metric->init. May be NULL for default configuration.
 * @retval 0 Success; @p p_metric->init(@p p_cfg) was called and the metric was appended.
 * @retval -EINVAL @p p_ctx or @p p_metric is NULL, or @p p_metric->snapshot is NULL.
 * @retval -EALREADY @p p_metric was already registered (init not called again).
 */
int nrf_edgeai_obsv_core_register(nrf_edgeai_obsv_core_t *p_ctx,
				  nrf_edgeai_obsv_metric_t *p_metric, const void *p_cfg);

/**
 * @brief Deregister a previously registered metric.
 *
 * Removes @p p_metric from the list. No-op if @p p_metric is not found.
 *
 * @param p_ctx    Initialized core.
 * @param p_metric Metric descriptor to remove.
 * @return 0 on success, -EINVAL if @p p_ctx or @p p_metric is NULL.
 */
int nrf_edgeai_obsv_core_deregister(nrf_edgeai_obsv_core_t *p_ctx,
				    nrf_edgeai_obsv_metric_t *p_metric);

/**
 * @brief Feed one inference result to all registered metrics.
 *
 * @param p_ctx   Initialized core.
 * @param p_probs Array of @c p_ctx->model.num_classes class probabilities.
 * @return 0 on success, -EINVAL if @p p_ctx or @p p_probs is NULL.
 */
int nrf_edgeai_obsv_core_update(nrf_edgeai_obsv_core_t *p_ctx, const float *p_probs);

/**
 * @brief Iterate over all registered metrics.
 *
 * Calls @p p_metric->finalize() then @p p_metric->snapshot() for each metric,
 * and invokes @p cb with the resulting snapshot. Iteration stops early if @p cb
 * returns false.
 *
 * @param p_ctx Initialized core.
 * @param cb    Callback invoked once per metric snapshot. Return false to stop early.
 * @param user  Opaque pointer forwarded to @p cb.
 * @return 0 on success, -EINVAL if @p p_ctx or @p cb is NULL.
 */
int nrf_edgeai_obsv_core_for_each_metric(nrf_edgeai_obsv_core_t *p_ctx,
					 nrf_edgeai_obsv_metric_cb_t cb, void *user);

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_CORE_H */

/**
 * @}
 */
