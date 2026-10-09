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
 * A metric may also report the configuration its counters were gathered with
 * (e.g. histogram ceilings) as a second row-major @c int32_t matrix of
 * @p config_rows x @p config_cols elements at @p config. The meaning and order
 * of the values is fixed per (@p metric_id, @p version). The core
 * zero-initializes the snapshot before calling snapshot(), so a metric with no
 * configuration leaves @p config NULL and nothing is emitted for it.
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
	/** @brief Number of rows in the config matrix (0 = no configuration reported). */
	uint16_t config_rows;
	/** @brief Number of columns in the config matrix. */
	uint16_t config_cols;
	/**
	 * @brief Row-major config matrix, @p config_rows x @p config_cols int32s.
	 *
	 * NULL when the metric reports no configuration. Like @p counts it points
	 * into the metric's @c priv storage; ownership remains with the metric.
	 */
	const int32_t *config;
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
 * Use @ref nrf_edgeai_obsv_metric_tm_create / @ref nrf_edgeai_obsv_metric_cpd_create
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
	NRF_EDGEAI_OBSV_METRIC_ID_MODEL_CERTAINTY_DESC = 1,
	NRF_EDGEAI_OBSV_METRIC_ID_CLASS_PRED_DIST = 2,
	NRF_EDGEAI_OBSV_METRIC_ID_TRANSITION_MATRIX = 3,
	NRF_EDGEAI_OBSV_METRIC_ID_MEL_ENERGY_DESC = 4,
	NRF_EDGEAI_OBSV_METRIC_ID_MEL_SPECTRAL_DESC = 5,
};

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
#define NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(n_classes)                                                \
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

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC)

/**
 * @brief Header (dimension/state fields) for the model certainty descriptor storage.
 *
 * Shared between the storage macro and the metric implementation so the layout
 * is defined in exactly one place. sizeof(_nrf_obsv_mcd_hdr_t) == 8, which keeps
 * the uint32_t counter array that immediately follows naturally aligned.
 * @c prev carries the previous inference's dominant class across updates (the
 * switching-rate state), sentinel 0xFFFF when no inference has been seen.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	uint16_t num_classes;
	uint16_t prev;
	uint8_t bin_num;
	uint8_t _pad[3];
} _nrf_obsv_mcd_hdr_t;

_Static_assert(sizeof(_nrf_obsv_mcd_hdr_t) == 8,
	       "Layout changed; update NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES and storage accessors");

/** @brief Rows of the model certainty descriptor matrix. */
#define NRF_EDGEAI_OBSV_MCD_NUM_ROWS 3

/**
 * @brief Minimum byte size of a model certainty descriptor storage buffer.
 *
 * The descriptor is a fixed @ref NRF_EDGEAI_OBSV_MCD_NUM_ROWS x bin_num matrix
 * (entropy histogram, top-2 margin histogram, stability counters), so storage
 * does not depend on @p n_classes; the parameter is accepted only for symmetry
 * with the other metric storage macros. Use with
 * @ref nrf_edgeai_obsv_metric_mcd_create. The buffer must be aligned to at least
 * @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(n_classes)                                               \
	(sizeof(_nrf_obsv_mcd_hdr_t) +                                                             \
	 (size_t)NRF_EDGEAI_OBSV_MCD_NUM_ROWS *                                                    \
		 CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM * sizeof(uint32_t))

/**
 * @brief Initialize a model certainty descriptor metric using caller-provided storage.
 *
 * Consumes the class-probability stream (@c NRF_EDGEAI_OBSV_SOURCE_PROBS). Per
 * inference it derives, in one pass, prediction uncertainty, decisiveness and
 * temporal stability, accumulating them into a @c 3 x bin_num matrix:
 *   - row 0: histogram of the normalized Shannon entropy @c H(p)/ln(N) over
 *     [0, 1] (uncertainty; high = uncertain / out-of-distribution);
 *   - row 1: histogram of the top-2 margin @c p_top1-p_top2 over [0, 1]
 *     (decisiveness; low = ambiguous near-tie);
 *   - row 2: stability counters
 *     @c [switches, comparisons, majority_frames, confident_switches], the rest
 *     of the row zero-padded. @c switches / @c comparisons is the off-device
 *     switching rate; @c majority_frames counts inferences with @c p_top1 > 0.5;
 *     @c confident_switches counts switches into a @c p_top1 > 0.5 winner.
 *
 * One bin count (@c CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM) is shared
 * by the entropy and margin histograms and sizes the stability-counter row.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_MCD_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_mcd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC */

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC)

/** @brief Layout of the config row reported by the mel energy descriptor. */
enum nrf_edgeai_obsv_med_config {
	/** @brief Lower scaling percentile p01, thousandths of a feature unit. */
	NRF_EDGEAI_OBSV_MED_CFG_SCALE_P01_MILLI = 0,
	/** @brief Upper scaling percentile p99, thousandths of a feature unit. */
	NRF_EDGEAI_OBSV_MED_CFG_SCALE_P99_MILLI = 1,
	/** @brief Number of config values (columns of the single config row). */
	NRF_EDGEAI_OBSV_MED_CFG_COUNT,
};

/**
 * @brief Header (dimension/scale fields) for the mel energy descriptor storage.
 *
 * Shared between the storage macro and the metric implementation.
 * sizeof(_nrf_obsv_med_hdr_t) == 12, which keeps the uint32_t counter array that
 * follows naturally aligned. @c cfg holds the configured percentile bounds
 * (p01 / p99, indexed by @ref nrf_edgeai_obsv_med_config) used to normalize
 * feature values into [0, 1]. They are thousandths of a feature unit, signed
 * (feature values, and so the percentiles, can be negative), so the snapshot
 * can point at them directly.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	int32_t cfg[NRF_EDGEAI_OBSV_MED_CFG_COUNT];
	uint16_t num_features;
	uint8_t bin_num;
	uint8_t _pad[1];
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
#define NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(n_features)                                              \
	(sizeof(_nrf_obsv_med_hdr_t) + (size_t)NRF_EDGEAI_OBSV_MED_NUM_ROWS *                      \
					       CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_BIN_NUM *    \
					       sizeof(uint32_t))

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
#define NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(n_features)                                              \
	(sizeof(_nrf_obsv_msd_hdr_t) + (size_t)NRF_EDGEAI_OBSV_MSD_NUM_ROWS *                      \
					       CONFIG_NRF_EDGEAI_OBSV_MEL_SPECTRAL_DESC_BIN_NUM *  \
					       sizeof(uint32_t))

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

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST)

/** @brief Layout of the config row reported by the class predictions distribution. */
enum nrf_edgeai_obsv_cpd_config {
	/** @brief Streak length saturating the top streak bin (STREAK_TOP_BIN). */
	NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOP = 0,
	/** @brief Flicker tolerance, in bridged frames (STREAK_TOL). */
	NRF_EDGEAI_OBSV_CPD_CFG_STREAK_TOL = 1,
	/** @brief Number of config values (columns of the single config row). */
	NRF_EDGEAI_OBSV_CPD_CFG_COUNT,
};

/**
 * @brief Header (dimension/config/state fields) for the class predictions distribution storage.
 *
 * Shared between the storage macro and the metric implementation so the layout
 * is defined in exactly one place. sizeof(_nrf_obsv_cpd_hdr_t) == 20, which keeps
 * the uint32_t counter array that immediately follows naturally aligned.
 * @c cfg holds the streak binning ceiling and flicker tolerance (indexed by
 * @ref nrf_edgeai_obsv_cpd_config) as @c int32_t so the snapshot can point at it
 * directly; the @c cur_* fields carry the in-progress streak state across updates
 * (a streak is recorded only when it ends). The @c alt_* fields track the trailing
 * run of one mismatching class inside the tolerance window, so that if tolerance is
 * exhausted those frames count toward the new streak instead of being lost.
 *
 * Not intended for direct use outside of the metric implementation.
 */
typedef struct {
	int32_t cfg[NRF_EDGEAI_OBSV_CPD_CFG_COUNT];
	uint16_t num_classes;
	uint16_t cur_class;
	uint16_t alt_class;
	uint8_t bin_num;
	uint8_t cur_len;
	uint8_t cur_miss;
	uint8_t alt_len;
	uint8_t _pad[2];
} _nrf_obsv_cpd_hdr_t;

_Static_assert(sizeof(_nrf_obsv_cpd_hdr_t) == 20,
	       "Layout changed; update NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES and storage accessors");

/** @brief Row-group count: the matrix is 2 x num_classes rows (probs, then streak). */
#define NRF_EDGEAI_OBSV_CPD_ROW_GROUPS 2

/**
 * @brief Minimum byte size of a class predictions distribution storage buffer for @p n_classes
 * classes.
 *
 * The matrix is @c 2*n_classes x bin_num (rows [0, n_classes) probability
 * distribution, rows [n_classes, 2*n_classes) streak distribution). Use with
 * @ref nrf_edgeai_obsv_metric_cpd_create to size a caller-supplied buffer. The
 * buffer must be aligned to at least @c sizeof(uint32_t) bytes.
 */
#define NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(n_classes)                                               \
	(sizeof(_nrf_obsv_cpd_hdr_t) + (size_t)NRF_EDGEAI_OBSV_CPD_ROW_GROUPS * (n_classes) *      \
					       CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM *    \
					       sizeof(uint32_t))

/**
 * @brief Initialize a class predictions distribution metric using caller-provided storage.
 *
 * Consumes the class-probability stream (@c NRF_EDGEAI_OBSV_SOURCE_PROBS) and, per
 * inference, feeds two per-class views into one @c 2*num_classes x bin_num matrix:
 *   - rows @c [0, num_classes): the probability distribution — each class's
 *     predicted probability binned uniformly over [0, 1]; every inference adds one
 *     sample to every class row, so each of these rows sums to the inference count;
 *   - rows @c [num_classes, 2*num_classes): the streak distribution — a per-class
 *     histogram of how many consecutive inferences the dominant class (argmax)
 *     stays that class. A streak is recorded when it ends; up to
 *     @c CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOL consecutive mismatching
 *     frames are bridged (not counted into the length, unless they turn out to
 *     start a new streak of another class, in which case they count toward that
 *     new streak). Streak lengths are
 *     binned uniformly over [1, TOP], lengths >= TOP saturating the top bin
 *     (@c CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_STREAK_TOP_BIN). These rows do not sum to the
 *     inference count.
 *
 * One bin count (@c CONFIG_NRF_EDGEAI_OBSV_CLASS_PRED_DIST_BIN_NUM) is shared by
 * both row groups. Merges the former probability distribution and class streak
 * distribution metrics.
 *
 * The caller allocates a buffer of at least @ref NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES
 * bytes, passes it here, then registers the metric with nrf_edgeai_obsv_core_register().
 *
 * @param metric    Metric descriptor to fill. Must not be NULL.
 * @param buf       Buffer of at least NRF_EDGEAI_OBSV_CPD_STORAGE_BYTES(n_classes)
 *                  bytes, aligned to at least @c sizeof(uint32_t). Must not be NULL.
 * @param n_classes Number of model output classes (> 0).
 */
void nrf_edgeai_obsv_metric_cpd_create(nrf_edgeai_obsv_metric_t *metric, void *buf,
				       uint16_t n_classes);

#endif /* CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST */

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_METRICS_H */

/**
 * @}
 */
