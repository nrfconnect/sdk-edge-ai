/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_obsv_metrics Observability metrics
 * @{
 * @ingroup nrf_edgeai_obsv
 *
 * @brief Metric descriptors, storage macros, and snapshot types.
 *
 */
#ifndef NRF_EDGEAI_OBSV_METRICS_H
#define NRF_EDGEAI_OBSV_METRICS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read-only view over one metric's accumulated counters.
 *
 * Produced by nrf_edgeai_obsv_metric_t::snapshot() and consumed by transport
 * layers (e.g. the Memfault CDR glue) to encode counters into whatever wire
 * format the transport chooses. @p counts points directly into the metric's
 * @c priv storage and remains valid for the lifetime of the metric instance.
 *
 * Counters are a row-major 2-D @c uint32_t matrix of @p num_rows x @p num_cols
 * elements. Metrics with 1-D data set @p num_cols to 1.
 *
 * Thread safety: snapshot() is not synchronized against concurrent
 * nrf_edgeai_obsv_core_update() calls. Callers must ensure @c update() does not
 * overlap @c snapshot() (e.g. hold the context lock for the duration of
 * for_each_metric() or encode()).
 */
typedef struct {
	/** @brief On-wire metric identifier. */
	uint32_t metric_id;
	/** @brief Metric payload version. */
	uint32_t version;
	/** @brief Number of rows in the counter matrix. */
	uint16_t num_rows;
	/** @brief Number of columns in the counter matrix. */
	uint16_t num_cols;
	/**
	 * @brief Row-major counter matrix, @p num_rows x @p num_cols uint32s.
	 *
	 * Points into the metric's @c priv storage. Ownership remains with the
	 * metric; the transport must not free it.
	 */
	const uint32_t *counts;
} nrf_edgeai_obsv_metric_snapshot_t;

/**
 * @brief Observability metric operation table and list node.
 *
 * Each metric implementation provides an instance of this structure containing
 * callbacks, a @p priv pointer to its own storage, and a list link pointer.
 * Metrics are registered into the observability context as a singly linked list.
 *
 * Use @ref nrf_edgeai_obsv_metric_tm_create / @ref nrf_edgeai_obsv_metric_pd_create
 * to initialize a metric descriptor with caller-provided storage.
 */
typedef struct nrf_edgeai_obsv_metric_s {
	/**
	 * @brief Initializes metric internal state.
	 * @param p_cfg Pointer to implementation-specific configuration.
	 * @param priv  Opaque per-instance storage pointer set by the define macro.
	 */
	void (*init)(const void *p_cfg, void *priv);

	/**
	 * @brief Consumes one inference result.
	 * @param p_probs Pointer to class-probability array for one inference.
	 * @param n       Number of entries in @p p_probs.
	 * @param priv    Opaque per-instance storage pointer.
	 */
	void (*update)(const float *p_probs, uint16_t n, void *priv);

	/**
	 * @brief Resets accumulated counters without touching configuration.
	 *
	 * Called by nrf_edgeai_obsv_reset() to zero counters while preserving
	 * any configuration set at registration time (e.g. custom bin edges).
	 * If NULL, the reset is a no-op for this metric (counters are not cleared).
	 *
	 * @param priv Opaque per-instance storage pointer.
	 */
	void (*clear)(void *priv);

	/**
	 * @brief Finalizes metric state before a snapshot is taken.
	 *
	 * May be NULL if the metric does not compute derived values.
	 *
	 * @param priv Opaque per-instance storage pointer.
	 */
	void (*finalize)(void *priv);

	/**
	 * @brief Populates a read-only view over the metric's counters.
	 *
	 * Implementations set every field of @p out and point @p out->counts
	 * directly at the live counter array inside @p priv. The pointer
	 * remains valid as long as @p priv is live (i.e. the metric instance
	 * exists). Callers must ensure update() cannot run concurrently while
	 * the snapshot is being read (e.g. hold the context lock for the
	 * duration of for_each_metric() or encode()).
	 *
	 * @param out  Output snapshot to populate.
	 * @param priv Opaque per-instance storage pointer.
	 */
	void (*snapshot)(nrf_edgeai_obsv_metric_snapshot_t *out, void *priv);

	/** @brief Opaque per-instance storage; set by the define macro. */
	void *priv;

	/** @brief Pointer to next metric in context list. */
	struct nrf_edgeai_obsv_metric_s *p_next;
} nrf_edgeai_obsv_metric_t;

/**
 * @brief Metric identifier values emitted by each metric's snapshot.
 */
enum nrf_edgeai_obsv_metric_id {
	NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX = 2,
	NRF_EDGEAI_OBSV_METRIC_ID_PROBS_DISTRIBUTION = 3,
};

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION)

/**
 * @brief Runtime configuration for probability distribution metric.
 *
 * Pass to nrf_edgeai_obsv_core_register() as the p_cfg argument.
 * NULL p_cfg gives uniform bins over [0, 1].
 */
typedef struct {
	/**
	 * Array of (CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM - 1) inner edge
	 * values in ascending order, e.g. {0.25, 0.5, 0.75} for 4 uniform bins over [0, 1].
	 * The boundary values 0.0 and 1.0 are implicit: all probability values below the
	 * first edge fall in bin 0; all values at or above the last edge fall in the last bin.
	 */
	float bin_edges[CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM - 1];
} nrf_obsv_probs_dist_cfg_t;

/**
 * @brief Header (dimension fields) for the probability distribution metric storage.
 *
 * Shared between the storage macro and the metric implementation so the layout
 * is defined in exactly one place. sizeof(_nrf_obsv_pd_hdr_t) == 4, which keeps
 * the float array that immediately follows naturally aligned.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_classes;
	uint8_t bin_num;
	uint8_t _pad[1];
} _nrf_obsv_pd_hdr_t;

_Static_assert(sizeof(_nrf_obsv_pd_hdr_t) == 4,
	       "Layout changed; update NRF_EDGEAI_OBSV_PD_STORAGE_BYTES and storage accessors");

/**
 * @brief Minimum byte size of a probability distribution storage buffer for @p n_classes classes.
 *
 * Use with @ref nrf_edgeai_obsv_metric_pd_create to size a caller-supplied buffer.
 * The buffer must be aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(n_classes)                                          \
	(sizeof(_nrf_obsv_pd_hdr_t)                                                          \
	 + ((size_t)CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM - 1U) * sizeof(float)  \
	 + (size_t)(n_classes) * CONFIG_NRF_EDGEAI_OBSV_PROBS_DISTRIBUTION_BIN_NUM           \
	   * sizeof(uint32_t))

/**
 * @brief Initialize a probability distribution metric using caller-provided storage.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_PD_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_pd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				      uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)

/**
 * @brief Header (dimension fields) for the transition matrix metric storage.
 *
 * Shared between the storage macro and the metric implementation.
 * sizeof(_nrf_obsv_tm_hdr_t) == 4, which keeps the uint32_t matrix that
 * immediately follows naturally aligned.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_classes;
	uint16_t prev;
} _nrf_obsv_tm_hdr_t;

_Static_assert(sizeof(_nrf_obsv_tm_hdr_t) == 4,
	       "Layout changed; update NRF_EDGEAI_OBSV_TM_STORAGE_BYTES and storage accessors");

/**
 * @brief Minimum byte size of a transition matrix storage buffer for @p n_classes classes.
 *
 * Use with @ref nrf_edgeai_obsv_metric_tm_create to size a caller-supplied buffer.
 * The buffer must be aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(n_classes) \
	(sizeof(_nrf_obsv_tm_hdr_t) + (size_t)(n_classes) * (size_t)(n_classes) * sizeof(uint32_t))

/**
 * @brief Initialize a transition matrix metric using caller-provided storage.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_TM_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_tm_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				      uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX */

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_METRICS_H */

/**
 * @}
 */
