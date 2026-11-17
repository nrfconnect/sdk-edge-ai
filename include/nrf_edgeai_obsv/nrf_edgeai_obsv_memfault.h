/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_obsv_memfault Memfault heartbeat transport
 * @{
 * @ingroup nrf_edgeai_obsv
 *
 * @brief Exports selected observability counters as Memfault heartbeat metrics.
 *
 */
#ifndef NRF_EDGEAI_OBSV_MEMFAULT_H
#define NRF_EDGEAI_OBSV_MEMFAULT_H

#include <memfault/metrics/metrics.h>

#ifdef __cplusplus
extern "C" {
#endif

struct nrf_edgeai_obsv_ctx;

/** @brief Memfault heartbeat keys for model identity and interval counters. */
struct nrf_edgeai_obsv_memfault_model_keys {
	/** @brief Key receiving the application-defined model identifier. */
	MemfaultMetricId model_id;
	/** @brief Key receiving the application-defined model version. */
	MemfaultMetricId model_version;
	/** @brief Key receiving the number of inferences in the heartbeat interval. */
	MemfaultMetricId num_inferences;
	/** @brief Key receiving the number of feature-vector updates in the heartbeat interval. */
	MemfaultMetricId num_feature_updates;
};

/** @brief Memfault heartbeat keys for one model certainty descriptor. */
struct nrf_edgeai_obsv_memfault_mcd_keys {
	/** @brief Key receiving the MCD payload version. */
	MemfaultMetricId version;
	/** @brief Keys receiving the normalized entropy histogram bins. */
	MemfaultMetricId entropy[CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM];
	/** @brief Keys receiving the top-2 margin histogram bins. */
	MemfaultMetricId top2_margin[CONFIG_NRF_EDGEAI_OBSV_MODEL_CERTAINTY_DESC_BIN_NUM];
	/** @brief Key receiving the dominant-class switch count. */
	MemfaultMetricId switches;
	/** @brief Key receiving the consecutive-pair comparison count. */
	MemfaultMetricId comparisons;
	/** @brief Key receiving the majority-frame count. */
	MemfaultMetricId majority_frames;
	/** @brief Key receiving the confident-switch count. */
	MemfaultMetricId confident_switches;
};

/** @brief Memfault heartbeat keys for one observability context. */
struct nrf_edgeai_obsv_memfault_keys {
	/** @brief Model identity and interval counter keys. */
	struct nrf_edgeai_obsv_memfault_model_keys model;
	/** @brief Model certainty descriptor keys. */
	struct nrf_edgeai_obsv_memfault_mcd_keys mcd;
};

/**
 * @brief Binds an observability context to Memfault heartbeat metrics.
 *
 * @p keys must remain valid for the lifetime of the application.
 *
 * @param ctx  Initialized observability context.
 * @param keys Heartbeat keys receiving the context's values.
 * @return 0 on success, negative errno on failure.
 */
int nrf_edgeai_obsv_memfault_init(struct nrf_edgeai_obsv_ctx *ctx,
				  const struct nrf_edgeai_obsv_memfault_keys *keys);

/**
 * @brief Copies observability counters into Memfault heartbeat metrics.
 *
 * Each context is snapshotted and reset while holding its lock, so every
 * heartbeat contains counters accumulated since the previous heartbeat without
 * losing an update between snapshot and reset. Contexts with no updates are
 * skipped, leaving their heartbeat keys unset.
 *
 * If Memfault event storage cannot hold a worst-case heartbeat, no keys are
 * set and no context is reset, so the counters carry over into the next
 * heartbeat that fits.
 *
 * This function is called automatically by Memfault immediately before a
 * heartbeat is serialized. It may also be called directly for testing.
 *
 * @retval 0       Success.
 * @retval -EINVAL No context has been registered.
 * @retval -ENOSPC Event storage is full; counters were held back.
 * @return First Memfault setter error encountered.
 */
int nrf_edgeai_obsv_memfault_collect(void);

#ifdef __cplusplus
}
#endif

#endif /* NRF_EDGEAI_OBSV_MEMFAULT_H */

/**
 * @}
 */
