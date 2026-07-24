#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Model-only OTA for a "Nordic EdgeAI Lab" solution exported with an Axon backend: a nrf_edgeai_t
# wrapper (input windowing, DSP feature pipeline, decode interfaces) around a compiled Axon model
# (nrf_axon_nn_compiled_model_s). Only the compiled Axon model is swappable; the wrapper stays
# compiled into the app.
#
# model_ota_axon_edgeai_wire(SOLUTION_ID <id> MODEL_SRC <abs-path-to-nrf_edgeai_user_model.c>
#                            HEADER <abs-path-to-nrf_edgeai_user_model_axon.h>
#                            PARTITION_NODELABEL <dt-nodelabel>
#                            [NAME <str>] [VERSION <x.y.z>]
#                            [PERSISTENT_VARS_CAP <n>] [MODEL_SYM <symbol>]
#                            [ALLOCATE_PACKED_OUTPUT])
#
# 1. Builds the compiled Axon model (the generated header's nrf_axon_nn_compiled_model_s object)
#    as a self-contained partition image, and app-owned RAM for its persistent vars / optional
#    packed-output buffer, via model_ota_axon_model() (see model_ota_axon.cmake) - exactly like a
#    pure-Axon model (e.g. person detection). Axon persistent vars are runtime feedback state
#    (see nrf_axon_nn_model_persistent_var_s) and can never live in flash, so the generated
#    header MUST declare them (and, if used, its packed-output buffer) via
#    NRF_AXON_MODEL_APP_STORAGE, exactly like a raw (non-wrapped) Axon export.
#
# 2. Generates model_ota_axon_edgeai_wired.c.in into
#    ${CMAKE_CURRENT_BINARY_DIR}/model_ota_axon_edgeai_wired_<SOLUTION_ID>.c (sets
#    MODEL_OTA_AXON_RUNTIME_WIRED, then #includes MODEL_SRC). Under that hook, the generated
#    nrf_edgeai_user_model.c skips its own #include of the generated Axon model header (so that
#    header's weights are never linked into the app image) and leaves model.instance.p_void
#    NULL. Builds a dedicated static library (default target ota_axon_edgeai_<SOLUTION_ID>) from
#    that file, defining nrf_edgeai_load_user_model_<SOLUTION_ID>(): loads the Axon model image
#    via model_image_load_axon() and patches it into model.instance.p_void.
#
# Models compiled directly into the app (CONFIG_MODEL_OTA_AXON=n) are unaffected: the generated
# nrf_edgeai_user_model.c includes the Axon model header itself, exactly as before.

include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/model_ota_axon.cmake)

get_filename_component(MODEL_OTA_ROOT ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)

function(model_ota_axon_edgeai_wire)
  cmake_parse_arguments(MO "ALLOCATE_PACKED_OUTPUT"
    "SOLUTION_ID;MODEL_SRC;HEADER;PARTITION_NODELABEL;NAME;VERSION;PERSISTENT_VARS_CAP;MODEL_SYM"
    "" ${ARGN})

  if(NOT MO_SOLUTION_ID OR NOT MO_MODEL_SRC OR NOT MO_HEADER OR NOT MO_PARTITION_NODELABEL)
    message(FATAL_ERROR "model_ota_axon_edgeai_wire requires SOLUTION_ID, MODEL_SRC, HEADER and "
                         "PARTITION_NODELABEL")
  endif()

  set(_wired_lib ota_axon_edgeai_${MO_SOLUTION_ID})
  if(TARGET ${_wired_lib})
    message(FATAL_ERROR "model_ota_axon_edgeai_wire: duplicate SOLUTION_ID ${MO_SOLUTION_ID}")
  endif()

  # Step 1: compiled Axon model -> partition image + app-owned persistent-vars/packed-output RAM.
  # TARGET drives the emitted <TARGET>_model_partition.hex / <TARGET>_model_image.bin names, so
  # use NAME (falling back to SOLUTION_ID) directly, same as a raw model_ota_axon_model() call.
  if(MO_NAME)
    set(_axon_target ${MO_NAME})
  else()
    set(_axon_target ${MO_SOLUTION_ID})
  endif()
  set(_axon_args
    TARGET ${_axon_target}
    HEADER ${MO_HEADER}
    PARTITION_NODELABEL ${MO_PARTITION_NODELABEL})
  if(MO_NAME)
    list(APPEND _axon_args NAME ${MO_NAME})
  endif()
  if(MO_VERSION)
    list(APPEND _axon_args VERSION ${MO_VERSION})
  endif()
  if(MO_PERSISTENT_VARS_CAP)
    list(APPEND _axon_args PERSISTENT_VARS_CAP ${MO_PERSISTENT_VARS_CAP})
  endif()
  if(MO_MODEL_SYM)
    list(APPEND _axon_args MODEL_SYM ${MO_MODEL_SYM})
  endif()
  if(MO_ALLOCATE_PACKED_OUTPUT)
    list(APPEND _axon_args ALLOCATE_PACKED_OUTPUT)
  endif()
  model_ota_axon_model(${_axon_args})

  # Step 2: wired nrf_edgeai_t wrapper (DSP pipeline, decode interfaces, loader) -> app.
  get_filename_component(model_dir ${MO_MODEL_SRC} DIRECTORY)
  get_filename_component(model_basename ${MO_MODEL_SRC} NAME)

  set(wired_tpl ${MODEL_OTA_ROOT}/src/model_ota_axon_edgeai_wired.c.in)
  set(wired_src ${CMAKE_CURRENT_BINARY_DIR}/model_ota_axon_edgeai_wired_${MO_SOLUTION_ID}.c)

  set(SOLUTION_ID ${MO_SOLUTION_ID})
  set(MODEL_SRC_BASENAME ${model_basename})
  configure_file(${wired_tpl} ${wired_src} @ONLY)

  add_library(${_wired_lib} STATIC ${wired_src})
  target_link_libraries(${_wired_lib} PRIVATE zephyr_interface)
  add_dependencies(${_wired_lib} zephyr_generated_headers)
  target_include_directories(${_wired_lib} PRIVATE ${model_dir} ${MODEL_OTA_ROOT}/src)
  set_source_files_properties(${wired_src}
                              TARGET_DIRECTORY ${_wired_lib}
                              PROPERTIES OBJECT_DEPENDS "${MO_MODEL_SRC}")
  target_link_libraries(app PRIVATE ${_wired_lib})
endfunction()
