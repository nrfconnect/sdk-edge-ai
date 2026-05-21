#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

zephyr_library()

zephyr_library_sources(nrf_edgeai_obsv_core.c)
zephyr_library_sources(nrf_edgeai_obsv.c)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_ENCODE nrf_edgeai_obsv_encode.c)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_METRIC_PROBS_DISTRIBUTION
  metrics/nrf_obsv_metric_probs_distribution.c
)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX
  metrics/nrf_obsv_metric_transition_matrix.c
)
