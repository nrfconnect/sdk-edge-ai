#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Detect whether ww_kws is being built with partition-resident models (APP_MODEL_OTA).
# Must run before find_package(Zephyr) / find_package(Sysbuild).

function(ww_kws_model_ota_enabled result)
  set(${result} FALSE PARENT_SCOPE)

  if(DEFINED CONFIG_APP_MODEL_OTA AND CONFIG_APP_MODEL_OTA)
    set(${result} TRUE PARENT_SCOPE)
    return()
  endif()

  if(DEFINED EXTRA_CONF_FILE AND EXTRA_CONF_FILE MATCHES "model_ota")
    set(${result} TRUE PARENT_SCOPE)
    return()
  endif()

  set(_ww_kws_sysbuild_conf "${CMAKE_CURRENT_BINARY_DIR}/zephyr/.config.sysbuild")
  if(EXISTS "${_ww_kws_sysbuild_conf}")
    file(READ "${_ww_kws_sysbuild_conf}" _ww_kws_sysbuild_conf_content)
    if(_ww_kws_sysbuild_conf_content MATCHES "(^|\n)CONFIG_APP_MODEL_OTA=y")
      set(${result} TRUE PARENT_SCOPE)
      return()
    endif()
    if(_ww_kws_sysbuild_conf_content MATCHES "(^|\n)CONFIG_UPDATEABLE_IMAGE_NUMBER=3")
      set(${result} TRUE PARENT_SCOPE)
      return()
    endif()
  endif()
endfunction()
