/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef WW_KWS_MODEL_OBSV_H_
#define WW_KWS_MODEL_OBSV_H_

/*
 * Shared model-observability module.
 *
 * Bundles one observability context together with the storage and descriptors of
 * every enabled built-in metric, so each model (wakeword, keyword spotting) keeps
 * a single `struct model_obsv` instance instead of open-coding buffers, metric
 * creation and Memfault binding in its own file. The set of metrics is identical
 * for every instance: whichever CONFIG_NRF_EDGEAI_OBSV_METRIC_* options are
 * enabled at build time are created and registered here.
 *
 * Metric storage is sized for the worst case (CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES
 * classes, MODEL_OBSV_MAX_FEATURES features), so one struct type fits a 2-class
 * wakeword context and a 12-class keyword-spotting context alike.
 *
 * The transport is Memfault (CBOR CDR); model_obsv_init() binds it. Snapshots are
 * drained by the Memfault packetizer (auto-collect on the system workqueue when
 * CONFIG_NRF_EDGEAI_OBSV_MEMFAULT_AUTO_COLLECT is set).
 */

#include <stdint.h>

#include <nrf_edgeai_obsv/nrf_edgeai_obsv.h>
#include <nrf_edgeai_obsv/nrf_edgeai_obsv_metrics.h>

/* Upper bound on the input-feature vector length used to size the FEATURES-metric
 * storage. Kept independent of the mel-energy MAX_FEATURES option so that the mel
 * spectral descriptor can be enabled without the mel energy descriptor.
 */
#if defined(CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES)
#define MODEL_OBSV_MAX_FEATURES CONFIG_NRF_EDGEAI_OBSV_MEL_ENERGY_DESC_MAX_FEATURES
#else
#define MODEL_OBSV_MAX_FEATURES 64
#endif

#define _MODEL_OBSV_WORDS(bytes) ((bytes) / sizeof(uint32_t))

/**
 * @brief One model's observability state: context + every enabled metric's
 *        descriptor and (uint32_t-aligned) counter storage.
 *
 * Declare one static instance per model and pass it to @ref model_obsv_init.
 */
struct model_obsv {
	nrf_edgeai_obsv_ctx_t ctx;

#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION)
	nrf_edgeai_obsv_metric_t pd;
	uint32_t pd_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_PD_STORAGE_BYTES(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX)
	nrf_edgeai_obsv_metric_t tm;
	uint32_t tm_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_TM_STORAGE_BYTES(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PREDICTION_SWITCHING_RATE)
	nrf_edgeai_obsv_metric_t psr;
	uint32_t psr_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_PSR_STORAGE_BYTES(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_ENTROPY_DIST)
	nrf_edgeai_obsv_metric_t ped;
	uint32_t ped_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_PED_STORAGE_BYTES(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_TOP2_MARGIN_DIST)
	nrf_edgeai_obsv_metric_t pmd;
	uint32_t pmd_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_PMD_STORAGE_BYTES(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_STREAK_DIST)
	nrf_edgeai_obsv_metric_t csd;
	uint32_t csd_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_CSD_STORAGE_BYTES(CONFIG_NRF_EDGEAI_OBSV_MAX_CLASSES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC)
	nrf_edgeai_obsv_metric_t med;
	uint32_t med_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_MED_STORAGE_BYTES(MODEL_OBSV_MAX_FEATURES))];
#endif
#if defined(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC)
	nrf_edgeai_obsv_metric_t msd;
	uint32_t msd_buf[_MODEL_OBSV_WORDS(
		NRF_EDGEAI_OBSV_MSD_STORAGE_BYTES(MODEL_OBSV_MAX_FEATURES))];
#endif
};

/**
 * @brief Initialize a model's observability: create and register every enabled
 *        metric, then bind the Memfault CDR transport.
 *
 * PROBS-stream metrics are sized/created for @c info->num_classes; FEATURES-stream
 * metrics for @p num_features. For a single-output (wakeword) model, set
 * @c info->num_classes to 2 and feed a synthesized [1 - p, p] vector to
 * @ref model_obsv_update_probs so the probability metrics see a real 2-class
 * distribution.
 *
 * @param o            Caller-owned, zeroed or static instance.
 * @param info         Model metadata; @c num_classes drives PROBS-metric storage.
 * @param num_features Input-feature vector length for FEATURES metrics (0 if the
 *                     model exposes no feature stream). Must not exceed
 *                     @ref MODEL_OBSV_MAX_FEATURES.
 * @return 0 on success, negative errno on failure.
 */
int model_obsv_init(struct model_obsv *o, const nrf_edgeai_obsv_model_info_t *info,
		    uint16_t num_features);

/** @brief Feed one class-probability vector (length @c info->num_classes) to the
 *         PROBS-stream metrics. Errors are logged, not returned.
 */
void model_obsv_update_probs(struct model_obsv *o, const float *probs);

/** @brief Feed one extracted input-feature vector to the FEATURES-stream metrics.
 *         Errors are logged, not returned.
 */
void model_obsv_update_features(struct model_obsv *o, const float *feats, uint16_t n);

#endif /* WW_KWS_MODEL_OBSV_H_ */
