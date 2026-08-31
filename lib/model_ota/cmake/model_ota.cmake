#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Model-only OTA build helpers — single include for application CMakeLists.txt.
#
#   model_ota_edgeai_neuton_model() — Edge AI Lab solution, Neuton backend
#   model_ota_edgeai_axon_model()   — Edge AI Lab solution, Axon backend
#   model_ota_axon_model()          — raw Axon model (no nrf_edgeai_t wrapper)
#   model_ota_context_finalize()    — export model_ota_context.json after link

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_context.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_edgeai_neuton.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/model_ota_axon.cmake)
