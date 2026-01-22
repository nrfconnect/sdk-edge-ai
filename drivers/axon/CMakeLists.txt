#
# Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
#
# The information contained herein is confidential property of Nordic Semiconductor ASA.
# The use, copying, transfer or disclosure of such information is prohibited except by
# express written agreement with Nordic Semiconductor ASA.
#

# use policy cmp0079
cmake_minimum_required(VERSION 3.13.0)

if (${NOT_A_ZEPHYR_BUILD})
  target_include_directories(nrf-axon-driver-interface INTERFACE include)

  target_link_libraries(nrf-axon-driver-interface
    INTERFACE
      nrf-axon-platform-interface
  ) 

  add_library(nrf-axon-driver STATIC EXCLUDE_FROM_ALL)
  set_target_properties(nrf-axon-driver PROPERTIES LINKER_LANGUAGE C)

  target_compile_definitions(nrf-axon-driver
	PRIVATE
	  AXON_SIMULATION=1
  )

  target_include_directories(nrf-axon-driver
    PRIVATE
      src
  )

  target_sources(nrf-axon-driver
    PRIVATE
      src/nrf_axon_nn_infer.c
      src/nrf_axon_nn_infer_test.c
	  src/nrf_axon_nn_op_extensions.c
	  ../compiler/src/nrf_axon_nn_extensions_compiler.c
  )

  target_link_libraries(nrf-axon-driver
    PUBLIC
      nrf-axon-driver-interface
    PRIVATE
      nrf-axon-platform-interface
  )
  target_link_libraries(nrf-axon-driver-interface
    INTERFACE
      nrf-axon-compiler-interface
  )
  # if this wasn't a full internal build, need to supply the nrf-axon-driver-internal library here.
  if (${CONFIG_NRF_AXON_FULL_SOURCE})
  else()
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
	endif()  

else()

	# add_library(nrf-axon-driver STATIC EXCLUDE_FROM_ALL)
	  zephyr_include_directories(
		  include
		  src
	  )

	zephyr_sources(
		src/nrf_axon_nn_infer.c
		src/nrf_axon_nn_infer_test.c
		src/nrf_axon_nn_op_extensions.c
	)
	if (${CONFIG_NRF_AXON_FULL_SOURCE})
	else()
		add_library(nrf-axon-driver-internal STATIC IMPORTED GLOBAL)
		if (${CONFIG_FPU})
			set(INTERNAL_DRIVER_LIB_NAME nrf-axon-driver-internal-fpu)
		else()
			set(INTERNAL_DRIVER_LIB_NAME nrf-axon-driver-internal)
		endif()
		
		message(STATUS  "linking imported library" ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_STATIC_LIBRARY_PREFIX}${INTERNAL_DRIVER_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX} )
		
		# cmake_system_processor will be either arm or riscv
		set_property(
			TARGET nrf-axon-driver-internal
			PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_STATIC_LIBRARY_PREFIX}${INTERNAL_DRIVER_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
		)
		
		zephyr_link_libraries(nrf-axon-driver-internal)
	endif()
endif()