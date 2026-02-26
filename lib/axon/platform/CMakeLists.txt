#
# Copyright (c) 2026 Nordic Semiconductor
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

message(STATUS ${CMAKE_CURRENT_SOURCE_DIR} " not zephyr? " ${NOT_A_ZEPHYR_BUILD})


if (${NOT_A_ZEPHYR_BUILD})
  if(NOT CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE)
    set(CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE 8000000) # make this big enough for the compiler to handle any model
  endif()
  if(NOT CONFIG_NRF_AXON_PSUM_BUFFER_SIZE)
    set(CONFIG_NRF_AXON_PSUM_BUFFER_SIZE 8000000) # make this big enough for the compiler to handle any model
  endif()
endif()


if (${NOT_A_ZEPHYR_BUILD})
  set(PLATFORM_DEFAULT_COMPILE_DEFS
    NRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE}
    NRF_AXON_PSUM_BUFFER_SIZE=${CONFIG_NRF_AXON_PSUM_BUFFER_SIZE}
  )

  add_library(nrf-axon-platform EXCLUDE_FROM_ALL)
	set_target_properties(nrf-axon-platform PROPERTIES LINKER_LANGUAGE C)

  add_library(nrf-axon-platform-interface INTERFACE)

  target_include_directories(nrf-axon-platform-interface
    INTERFACE
      ../../../include/axon
  )

  target_sources(nrf-axon-platform
    PRIVATE
      src/nrf_axon_logging.c
      src/nrf_axon_vector_compare.c
      src/simulator/nrf_axon_platform_simulator.c
  )

  message(STATUS "CMAKE_SYSTEM_NAME: ${CMAKE_SYSTEM_NAME}")

  if(CMAKE_SYSTEM_NAME STREQUAL "Generic")
    # no operating system platform specific source files to include
  elseif (MSVC OR WIN32)
    target_sources(nrf-axon-platform
      PRIVATE
        src/simulator/nrf_axon_platform_simulator_windows.c
    )
  elseif(UNIX)
    if(APPLE)
      target_sources(nrf-axon-platform
      PRIVATE
        src/simulator/nrf_axon_platform_simulator_macos.c
      )
    else()
      target_sources(nrf-axon-platform
        PRIVATE
          src/simulator/nrf_axon_platform_simulator_linux.c
      )
    endif()
  else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
  endif()

  target_link_libraries(nrf-axon-platform
    PRIVATE
      nrf-axon-platform-interface
      nrf-axon-driver-interface
  )

  target_compile_definitions(nrf-axon-platform-interface
    INTERFACE
      ${PLATFORM_DEFAULT_COMPILE_DEFS}
  )

  # CMAKE_HOST_SYSTEM_NAME will be Linux, Windows, or Darwin (for MacOS)
  message(STATUS  "linking imported library " ${CMAKE_CURRENT_SOURCE_DIR}/bin/simulator/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_STATIC_LIBRARY_PREFIX}nrf-axon-simulator-internal${CMAKE_STATIC_LIBRARY_SUFFIX})
  add_library(nrf-axon-simulator-internal STATIC IMPORTED GLOBAL)
  # Add the dependency for the pre-built library
  if(UNIX AND NOT APPLE)
    target_link_libraries(nrf-axon-simulator-internal INTERFACE m)
  endif()
  set_property(
    TARGET nrf-axon-simulator-internal
    PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/simulator/${CMAKE_HOST_SYSTEM_NAME}/${CMAKE_STATIC_LIBRARY_PREFIX}nrf-axon-simulator-internal${CMAKE_STATIC_LIBRARY_SUFFIX}
  )

else()
  # else a zephyr build
  message(STATUS "CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE " ${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE})
  message(STATUS "CONFIG_NRF_AXON_PSUM_BUFFER_SIZE " ${CONFIG_NRF_AXON_PSUM_BUFFER_SIZE})

  if(NOT CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE)
    # interlayer buffer is not in use
	  set(CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE 0)
  endif()
  if(NOT CONFIG_NRF_AXON_PSUM_BUFFER_SIZE)
    # psum buffer is not in use
	  set(CONFIG_NRF_AXON_PSUM_BUFFER_SIZE 0)
  endif()

  zephyr_sources(
    src/nrf_axon_logging.c
    src/nrf_axon_vector_compare.c
    src/zephyr/nrf_axon_platform_zephyr.c
  )
  set(PLATFORM_DEFAULT_COMPILE_DEFS
    NRF_AXON_INTERLAYER_BUFFER_SIZE=${CONFIG_NRF_AXON_INTERLAYER_BUFFER_SIZE}
    NRF_AXON_PSUM_BUFFER_SIZE=${CONFIG_NRF_AXON_PSUM_BUFFER_SIZE}
  )

  zephyr_compile_definitions(
      ${PLATFORM_DEFAULT_COMPILE_DEFS}
      NRF_AXON_INTERLAYER_BUFFER_MEMREGION="\""${CONFIG_NRF_AXON_INTERLAYER_BUFFER_MEMREGION_NAME}"\""
  )

endif()
