#
# Copyright (c) 2025 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

target_sources_ifdef(CONFIG_ML_APP_ML_RUNNER_STUB
  app PRIVATE
  stub_runner.c
)

target_sources_ifdef(CONFIG_ML_APP_ML_RUNNER_EI
  app PRIVATE
  ei_runner.c
)

target_sources_ifdef(CONFIG_ML_APP_ML_RUNNER_NEUTON
  app PRIVATE
  neuton_runner.c
)

if(CONFIG_ML_APP_ML_RUNNER_NEUTON)
  include(neuton.cmake)
endif()
