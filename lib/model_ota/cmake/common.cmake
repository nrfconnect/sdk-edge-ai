#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# Shared helpers for model_ota build steps.

include_guard(GLOBAL)

set(MODEL_OTA_TOOLS_DIR ${ZEPHYR_EDGE_AI_MODULE_DIR}/scripts/model_ota
    CACHE INTERNAL "edge-ai model_ota host tools")
set(MODEL_OTA_LIB_DIR ${ZEPHYR_EDGE_AI_MODULE_DIR}/lib/model_ota
    CACHE INTERNAL "edge-ai model_ota library sources")

# The application's own C flags, for compiling a throwaway probe translation unit with exactly the
# firmware's view of the headers and Kconfig.
function(model_ota_zephyr_c_compile_flags OUT_VAR)
  zephyr_get_include_directories_for_lang(C inc)
  zephyr_get_system_include_directories_for_lang(C sys)
  zephyr_get_compile_definitions_for_lang(C def)
  zephyr_get_compile_options_for_lang(C opt)
  set(${OUT_VAR} ${opt} ${inc} ${sys} ${def} PARENT_SCOPE)
endfunction()
