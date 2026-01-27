#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

# use policy cmp0079
cmake_minimum_required(VERSION 3.13.0)

add_subdirectory(platform)

if (${NOT_A_ZEPHYR_BUILD})
  # CMAKE_HOST_SYSTEM_NAME will be Linux, Windows, or Darwin (for MacOS)
  message(STATUS  "linking imported library " ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_STATIC_LIBRARY_PREFIX}nrf-axon-driver-internal${CMAKE_STATIC_LIBRARY_SUFFIX})
  add_library(nrf-axon-driver-internal STATIC IMPORTED GLOBAL)
  set_property(
    TARGET nrf-axon-driver-internal
    PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_STATIC_LIBRARY_PREFIX}nrf-axon-driver-internal${CMAKE_STATIC_LIBRARY_SUFFIX}
  )
  message(STATUS  "linking imported library " ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_STATIC_LIBRARY_PREFIX}nrf-axon-nn-compiler-internal${CMAKE_STATIC_LIBRARY_SUFFIX})
  add_library(nrf-axon-nn-compiler-internal STATIC IMPORTED GLOBAL)
  set_property(
    TARGET nrf-axon-nn-compiler-internal
    PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_STATIC_LIBRARY_PREFIX}nrf-axon-nn-compiler-internal${CMAKE_STATIC_LIBRARY_SUFFIX}
  )

else()

  add_library(nrf-axon-driver-internal STATIC IMPORTED GLOBAL)
  if (${CONFIG_FPU})
    set(INTERNAL_DRIVER_LIB_NAME nrf-axon-driver-internal-fpu)
  else()
    set(INTERNAL_DRIVER_LIB_NAME nrf-axon-driver-internal)
  endif()

  message(STATUS  "linking imported library" ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_STATIC_LIBRARY_PREFIX}${INTERNAL_DRIVER_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX} )

  if(NOT CONFIG_BOARD_NATIVE_SIM)
    # cmake_system_processor will be either arm or riscv
    set_property(
      TARGET nrf-axon-driver-internal
      PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_STATIC_LIBRARY_PREFIX}${INTERNAL_DRIVER_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
  else()
    set_property(
      TARGET nrf-axon-driver-internal
      PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_STATIC_LIBRARY_PREFIX}nrf-axon-driver-internal${CMAKE_STATIC_LIBRARY_SUFFIX}
    )
  endif()

  zephyr_link_libraries(nrf-axon-driver-internal)
endif()
