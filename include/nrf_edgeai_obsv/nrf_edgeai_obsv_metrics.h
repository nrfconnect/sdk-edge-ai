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
 * nrf_edgeai_obsv_core_update_probs() calls. Callers must ensure @c update() does not
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
 * @brief Data stream a metric consumes.
 *
 * An observability context carries more than one stream: the model output
 * (class-probability vector) and, optionally, the extracted input features fed
 * to the model. Each metric declares which stream it consumes via
 * @ref nrf_edgeai_obsv_metric_s::source, and the core routes an update only to
 * metrics whose source matches the fed stream. Probabilities are fed with
 * @ref nrf_edgeai_obsv_update_probs; features with @ref nrf_edgeai_obsv_update_features.
 */
enum nrf_edgeai_obsv_source {
	/** @brief Model output: class-probability vector (length == num_classes). */
	NRF_EDGEAI_OBSV_SOURCE_PROBS = 0, /* metric fed via nrf_edgeai_obsv_update_probs */
	/** @brief Model input: extracted feature vector (length supplied per update). */
	NRF_EDGEAI_OBSV_SOURCE_FEATURES = 1,
};

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
	 * @brief Consumes one data vector from the metric's source stream.
	 *
	 * The core invokes this only for the stream the metric declared via
	 * @ref nrf_edgeai_obsv_metric_s::source. The class-probability vector for
	 * @c NRF_EDGEAI_OBSV_SOURCE_PROBS metrics, or the extracted input-feature
	 * vector for @c NRF_EDGEAI_OBSV_SOURCE_FEATURES metrics.
	 *
	 * @param p_data Pointer to the input vector for one update: class
	 *               probabilities (PROBS) or extracted features (FEATURES),
	 *               per the metric's source.
	 * @param n      Number of entries in @p p_data: @c num_classes for PROBS,
	 *               or the length passed to @ref nrf_edgeai_obsv_update_features
	 *               for FEATURES.
	 * @param priv   Opaque per-instance storage pointer.
	 */
	void (*update)(const float *p_data, uint16_t n, void *priv);

	/**
	 * @brief Resets accumulated counters without touching configuration.
	 *
	 * Called by nrf_edgeai_obsv_reset() to zero counters while preserving
	 * any configuration set at registration time (e.g. the custom bin edges
	 * of the probability distribution metric).
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

	/**
	 * @brief Data stream this metric consumes (@ref nrf_edgeai_obsv_source).
	 *
	 * Set by the metric's @c *_create() helper: @c NRF_EDGEAI_OBSV_SOURCE_PROBS
	 * for the probability metrics, @c NRF_EDGEAI_OBSV_SOURCE_FEATURES for the
	 * mel descriptor metrics. The core dispatches an update to this metric only
	 * when the fed stream matches this value.
	 */
	enum nrf_edgeai_obsv_source source;

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
	NRF_EDGEAI_OBSV_METRIC_ID_PREDICTION_SWITCHING_RATE = 4,
	NRF_EDGEAI_OBSV_METRIC_ID_PROBS_ENTROPY_DIST = 5,
	NRF_EDGEAI_OBSV_METRIC_ID_PROBS_TOP2_MARGIN_DIST = 6,
	NRF_EDGEAI_OBSV_METRIC_ID_MEL_ENERGY_DESC = 7,
	NRF_EDGEAI_OBSV_METRIC_ID_MEL_SPECTRAL_DESC = 8,
	NRF_EDGEAI_OBSV_METRIC_ID_CLASS_STREAK_DIST = 9,
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

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE)

/**
 * @brief Header (dimension/state fields) for the prediction switching rate storage.
 *
 * Shared between the storage macro and the metric implementation.
 * sizeof(_nrf_obsv_psr_hdr_t) == 4, which keeps the uint32_t counters that
 * immediately follow naturally aligned.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_classes;
	uint16_t prev;
} _nrf_obsv_psr_hdr_t;

_Static_assert(sizeof(_nrf_obsv_psr_hdr_t) == 4,
	       "Layout changed; update NRF_EDGEAI_OBSV_PSR_STORAGE_BYTES and storage accessors");

/**
 * @brief Minimum byte size of a prediction switching rate storage buffer.
 *
 * The metric exposes a fixed 1 x 2 row of @c uint32_t counters
 * (switches, comparisons), so the storage size does not depend on @p n_classes;
 * the parameter is accepted only for symmetry with the other metric storage
 * macros. Use with @ref nrf_edgeai_obsv_metric_psr_create to size a
 * caller-supplied buffer. The buffer must be aligned to at least
 * @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_PSR_STORAGE_BYTES(n_classes) \
	(sizeof(_nrf_obsv_psr_hdr_t) + 2U * sizeof(uint32_t))

/**
 * @brief Initialize a prediction switching rate metric using caller-provided storage.
 *
 * Tracks temporal instability: the metric counts how often the dominant class
 * (argmax of the probability vector) changes between consecutive inferences.
 * It exports two raw counters as a 1 x 2 row, @c [switches, comparisons], from
 * which the rate is derived off-device:
 * @c SwitchRate = switches / comparisons = (1 / (N - 1)) * sum I(y_t != y_{t-1}).
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_PSR_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_PSR_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_psr_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST)

/**
 * @brief Header (dimension fields) for the probability entropy distribution storage.
 *
 * Shared between the storage macro and the metric implementation so the layout
 * is defined in exactly one place. sizeof(_nrf_obsv_ped_hdr_t) == 4, which keeps
 * the uint32_t counter array that immediately follows naturally aligned.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_classes;
	uint8_t bin_num;
	uint8_t _pad[1];
} _nrf_obsv_ped_hdr_t;

_Static_assert(sizeof(_nrf_obsv_ped_hdr_t) == 4,
	       "Layout changed; update NRF_EDGEAI_OBSV_PED_STORAGE_BYTES and storage accessors");

/**
 * @brief Minimum byte size of a probability entropy distribution storage buffer.
 *
 * Entropy is a single scalar per inference, so the histogram is one row of
 * @c CONFIG_NRF_EDGEAI_OBSV_PROBS_ENTROPY_DIST_BIN_NUM bins regardless of class
 * count; @p n_classes is accepted only for symmetry with the other metric
 * storage macros. Use with @ref nrf_edgeai_obsv_metric_ped_create. The buffer
 * must be aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_PED_STORAGE_BYTES(n_classes)                                          \
	(sizeof(_nrf_obsv_ped_hdr_t)                                                          \
	 + (size_t)CONFIG_NRF_EDGEAI_OBSV_PROBS_ENTROPY_DIST_BIN_NUM * sizeof(uint32_t))

/**
 * @brief Initialize a probability entropy distribution metric using caller-provided storage.
 *
 * Measures prediction uncertainty: per inference it computes the Shannon entropy
 * @c H(p)=-sum p_i*ln(p_i) of the probability vector, normalizes it to [0, 1] by
 * the maximum entropy @c ln(num_classes), and bins it into a 1 x bin_num
 * histogram. High entropy indicates uncertain or out-of-distribution inputs.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_PED_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_PED_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_ped_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST)

/**
 * @brief Header (dimension fields) for the probability top-2 margin distribution storage.
 *
 * Shared between the storage macro and the metric implementation so the layout
 * is defined in exactly one place. sizeof(_nrf_obsv_pmd_hdr_t) == 4, which keeps
 * the uint32_t counter array that immediately follows naturally aligned.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_classes;
	uint8_t bin_num;
	uint8_t _pad[1];
} _nrf_obsv_pmd_hdr_t;

_Static_assert(sizeof(_nrf_obsv_pmd_hdr_t) == 4,
	       "Layout changed; update NRF_EDGEAI_OBSV_PMD_STORAGE_BYTES and storage accessors");

/**
 * @brief Minimum byte size of a probability top-2 margin distribution storage buffer.
 *
 * The margin is a single scalar per inference, so the histogram is one row of
 * @c CONFIG_NRF_EDGEAI_OBSV_PROBS_TOP2_MARGIN_DIST_BIN_NUM bins regardless of
 * class count; @p n_classes is accepted only for symmetry with the other metric
 * storage macros. Use with @ref nrf_edgeai_obsv_metric_pmd_create. The buffer
 * must be aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_PMD_STORAGE_BYTES(n_classes)                                            \
	(sizeof(_nrf_obsv_pmd_hdr_t)                                                            \
	 + (size_t)CONFIG_NRF_EDGEAI_OBSV_PROBS_TOP2_MARGIN_DIST_BIN_NUM * sizeof(uint32_t))

/**
 * @brief Initialize a probability top-2 margin distribution metric using caller-provided storage.
 *
 * Measures how decisive each prediction is: per inference it computes the margin
 * @c margin=p_top1-p_top2 between the two largest class probabilities and bins it
 * into a 1 x bin_num histogram over [0, 1]. A low margin flags ambiguous
 * predictions even when the dominant probability is high.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_PMD_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_PMD_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_pmd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC)

/**
 * @brief Header (dimension/scale fields) for the mel energy descriptor storage.
 *
 * Shared between the storage macro and the metric implementation.
 * sizeof(_nrf_obsv_med_hdr_t) == 12, which keeps the uint32_t counter array that
 * follows naturally aligned. @c scale_min / @c scale_max are the configured
 * percentile bounds (p01 / p99) used to normalize feature values into [0, 1].
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_features;
	uint8_t bin_num;
	uint8_t _pad[1];
	float scale_min;
	float scale_max;
} _nrf_obsv_med_hdr_t;

_Static_assert(sizeof(_nrf_obsv_med_hdr_t) == 12,
	       "Layout changed; update NRF_EDGEAI_OBSV_MED_STORAGE_BYTES and storage accessors");

/** @brief Descriptor rows: mean energy, max energy, dynamic range, floor ratio. */
#define NRF_EDGEAI_OBSV_MED_NUM_ROWS 4

/**
 * @brief Minimum byte size of a mel energy descriptor storage buffer.
 *
 * The descriptor is @ref NRF_EDGEAI_OBSV_MED_NUM_ROWS rows of
 * @c CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM bins; storage does not depend
 * on @p n_features (accepted only for symmetry with the other metric storage
 * macros). Use with @ref nrf_edgeai_obsv_metric_med_create. The buffer must be
 * aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(n_features)                                           \
	(sizeof(_nrf_obsv_med_hdr_t)                                                            \
	 + (size_t)NRF_EDGEAI_OBSV_MED_NUM_ROWS                                                 \
	   * CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM * sizeof(uint32_t))

/**
 * @brief Initialize a mel energy descriptor metric using caller-provided storage.
 *
 * Consumes the input-feature stream (@c NRF_EDGEAI_OBSV_SOURCE_FEATURES). Each
 * feature value is normalized into [0, 1] against the configured percentile
 * range [p01, p99] (@c CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_SCALE_P01_MILLI /
 * @c _SCALE_P99_MILLI, in thousandths of a feature unit). The metric then derives
 * four per-frame statistics from the normalized vector — mean energy, max energy,
 * dynamic range (q95 - q05) — plus the floor-bin ratio (fraction of raw bins
 * <= 0) — and accumulates each into its own [0, 1] histogram row.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_MED_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric     Metric descriptor to fill. Must not be NULL.
 * @param buf        Buffer of at least NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(n_features)
 *                   bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_features Mel feature vector length (> 0), used to validate updates.
 */
void nrf_edgeai_obsv_metric_med_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_features);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC)

/**
 * @brief Header (dimension fields) for the mel spectral descriptor storage.
 *
 * Shared between the storage macro and the metric implementation.
 * sizeof(_nrf_obsv_msd_hdr_t) == 4, which keeps the uint32_t counter array that
 * follows naturally aligned. The metric stores no scale state: every row is a
 * scale-invariant shape statistic already in [0, 1].
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_features;
	uint8_t bin_num;
	uint8_t _pad[1];
} _nrf_obsv_msd_hdr_t;

_Static_assert(sizeof(_nrf_obsv_msd_hdr_t) == 4,
	       "Layout changed; update NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES and storage accessors");

/** @brief Descriptor rows: low/mid/high ratio, centroid, spread, entropy, flatness, contrast. */
#define NRF_EDGEAI_OBSV_MSD_NUM_ROWS 8

/**
 * @brief Minimum byte size of a mel spectral descriptor storage buffer.
 *
 * The descriptor is @ref NRF_EDGEAI_OBSV_MSD_NUM_ROWS rows of
 * @c CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM bins; storage does not
 * depend on @p n_features (accepted only for symmetry with the other metric
 * storage macros). Use with @ref nrf_edgeai_obsv_metric_msd_create. The buffer
 * must be aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(n_features)                                           \
	(sizeof(_nrf_obsv_msd_hdr_t)                                                            \
	 + (size_t)NRF_EDGEAI_OBSV_MSD_NUM_ROWS                                                 \
	   * CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM * sizeof(uint32_t))

/**
 * @brief Initialize a mel spectral descriptor metric using caller-provided storage.
 *
 * Consumes the input-feature stream (@c NRF_EDGEAI_OBSV_SOURCE_FEATURES). For
 * each feature vector it derives eight scale-invariant spectral-shape statistics
 * — low/mid/high band energy ratios, spectral centroid, spread, entropy,
 * flatness, and contrast — each normalized to [0, 1] and accumulated into its own
 * histogram row. No amplitude calibration is required.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric     Metric descriptor to fill. Must not be NULL.
 * @param buf        Buffer of at least NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(n_features)
 *                   bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_features Mel feature vector length (> 0), used to validate updates.
 */
void nrf_edgeai_obsv_metric_msd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_features);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_STREAK_DIST)

/**
 * @brief Header (dimension/config/state fields) for the class streak distribution storage.
 *
 * Shared between the storage macro and the metric implementation so the layout
 * is defined in exactly one place. sizeof(_nrf_obsv_csd_hdr_t) == 12, which keeps
 * the uint32_t counter array that immediately follows naturally aligned.
 * @c top and @c tolerance are the configured binning ceiling and flicker
 * tolerance; the @c cur_* fields carry the in-progress streak state across
 * updates (a streak is recorded only when it ends).
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_classes;
	uint16_t cur_class;
	uint8_t bin_num;
	uint8_t top;
	uint8_t tolerance;
	uint8_t cur_len;
	uint8_t cur_miss;
	uint8_t _pad[3];
} _nrf_obsv_csd_hdr_t;

_Static_assert(sizeof(_nrf_obsv_csd_hdr_t) == 12,
	       "Layout changed; update NRF_EDGEAI_OBSV_CSD_STORAGE_BYTES and storage accessors");

/**
 * @brief Minimum byte size of a class streak distribution storage buffer for @p n_classes classes.
 *
 * Use with @ref nrf_edgeai_obsv_metric_csd_create to size a caller-supplied buffer.
 * The buffer must be aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_CSD_STORAGE_BYTES(n_classes)                                          \
	(sizeof(_nrf_obsv_csd_hdr_t)                                                          \
	 + (size_t)(n_classes) * CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_BIN_NUM             \
	   * sizeof(uint32_t))

/**
 * @brief Initialize a class streak distribution metric using caller-provided storage.
 *
 * Per class, accumulates a histogram of streak lengths: the number of consecutive
 * inferences whose dominant class (argmax) is that class. A streak is recorded
 * into its class's histogram row when it ends. Up to
 * @c CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOLERANCE consecutive mismatching
 * frames are bridged without ending the streak (and are not counted into its
 * length); more consecutive mismatches than that end it. Streak lengths are binned
 * uniformly over [1, TOP], with lengths >= TOP saturating the top bin
 * (@c CONFIG_NRF_EDGEAI_OBSV_CLASS_STREAK_DIST_TOP). Consumes the class-probability
 * stream (@c NRF_EDGEAI_OBSV_SOURCE_PROBS).
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_CSD_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_CSD_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_csd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_STREAK_DIST */

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_METRICS_H */

/**
 * @}
 */
