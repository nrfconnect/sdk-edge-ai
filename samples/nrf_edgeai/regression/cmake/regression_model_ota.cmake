#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Detect whether regression is being built with a partition-resident model (APP_MODEL_OTA).
# Must run before find_package(Zephyr) / find_package(Sysbuild).

function(regression_model_ota_enabled result)
  set(${result} FALSE PARENT_SCOPE)

  if(DEFINED CONFIG_APP_MODEL_OTA AND CONFIG_APP_MODEL_OTA)
    set(${result} TRUE PARENT_SCOPE)
    return()
  endif()

  if(DEFINED EXTRA_CONF_FILE AND EXTRA_CONF_FILE MATCHES "model_ota")
    set(${result} TRUE PARENT_SCOPE)
    return()
  endif()

  set(_regression_sysbuild_conf "${CMAKE_CURRENT_BINARY_DIR}/zephyr/.config.sysbuild")
  if(EXISTS "${_regression_sysbuild_conf}")
    file(READ "${_regression_sysbuild_conf}" _regression_sysbuild_conf_content)
    if(_regression_sysbuild_conf_content MATCHES "(^|\n)CONFIG_APP_MODEL_OTA=y")
      set(${result} TRUE PARENT_SCOPE)
      return()
    endif()
    if(_regression_sysbuild_conf_content MATCHES "(^|\n)CONFIG_UPDATEABLE_IMAGE_NUMBER=2")
      set(${result} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
endfunction()

function(regression_model_ota_board_overlay result)
  if(DEFINED BOARD AND BOARD MATCHES "nrf54lm20a")
    set(${result} nrf54lm20dk_nrf54lm20a_cpuapp_model_ota.overlay PARENT_SCOPE)
  else()
    set(${result} nrf54lm20dk_nrf54lm20b_cpuapp_model_ota.overlay PARENT_SCOPE)
  endif()
endfunction()
