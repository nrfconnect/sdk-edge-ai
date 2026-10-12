#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

zephyr_library()

zephyr_library_sources(nrf_edgeai_obsv_core.c)
zephyr_library_sources(nrf_edgeai_obsv.c)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_ENCODE nrf_edgeai_obsv_encode.c)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_DIST_BINNING
  metrics/nrf_obsv_dist_binning.c
)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_METRIC_TRANSITION_MATRIX
  metrics/nrf_obsv_metric_transition_matrix.c
)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_METRIC_MODEL_CERTAINTY_DESC
  metrics/nrf_obsv_metric_model_certainty_desc.c
)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_METRIC_CLASS_PRED_DIST
  metrics/nrf_obsv_metric_class_pred_dist.c
)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_ENERGY_DESC
  metrics/nrf_obsv_metric_mel_energy_desc.c
)

zephyr_library_sources_ifdef(CONFIG_NRF_EDGEAI_OBSV_METRIC_MEL_SPECTRAL_DESC
  metrics/nrf_obsv_metric_mel_spectral_desc.c
)
