#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})

project(test_nrf_edgeai_obsv)

target_sources(app PRIVATE
  src/capture.c
  src/suite_core.c
  src/suite_custom_metric.c
  src/suite_iteration.c
  src/suite_payload.c
  src/suite_user_storage.c
  src/suite_pred_switch_rate.c
  src/suite_probs_entropy_dist.c
  src/suite_probs_top2_margin_dist.c
  src/suite_class_streak_dist.c
  src/suite_mel_energy_desc.c
  src/suite_mel_spectral_desc.c
)

# Encoder CDDL conformance suite — only built when the encoder is enabled.
if(CONFIG_NRF_EDGEAI_OBSV_ENCODE)
  set(OBSV_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/nrf_edgeai_obsv)

  # Generate a strict CBOR decoder for obsv-cdr from the authoritative schema
  # so the test can validate nrf_edgeai_obsv_encode_multi_cbor() output without
  # any transport dependency. Regenerated on every configure; CDDL drift fails
  # the build.
  set(OBSV_CDDL ${OBSV_LIB}/obsv.cddl)
  set(CDDL_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/cddl_gen)
  set(CDDL_GEN_C ${CDDL_GEN_DIR}/cddl_decode.c)
  set(CDDL_GEN_H ${CDDL_GEN_DIR}/cddl_decode.h)
  set(CDDL_GEN_TYPES_H ${CDDL_GEN_DIR}/cddl_decode_types.h)

  find_program(ZCBOR_EXECUTABLE zcbor REQUIRED)

  add_custom_command(
    OUTPUT ${CDDL_GEN_C} ${CDDL_GEN_H} ${CDDL_GEN_TYPES_H}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CDDL_GEN_DIR}
    COMMAND ${ZCBOR_EXECUTABLE} code
            -c ${OBSV_CDDL}
            -d
            -t obsv-list
            --output-c ${CDDL_GEN_C}
            --output-h ${CDDL_GEN_H}
            --output-h-types ${CDDL_GEN_TYPES_H}
            --default-max-qty 8
    DEPENDS ${OBSV_CDDL}
    COMMENT "Generating CBOR decoder from ${OBSV_CDDL}"
    VERBATIM
  )

  add_custom_target(cddl_gen DEPENDS ${CDDL_GEN_C} ${CDDL_GEN_H} ${CDDL_GEN_TYPES_H})
  add_dependencies(app cddl_gen)

  target_include_directories(app PRIVATE ${CDDL_GEN_DIR})

  target_sources(app PRIVATE
    src/suite_encoder.c
    ${CDDL_GEN_C}
  )
endif()
