#
# Copyright (c) 2026 Nordic Semiconductor ASA
#
# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#

###############################################################################
# Directory Configuration
###############################################################################
set(EDGE_IMPULSE_DIR ${CMAKE_BINARY_DIR}/edge_impulse)
set(EDGE_IMPULSE_SOURCE_DIR ${EDGE_IMPULSE_DIR}/src/edge_impulse_project)
set(EDGE_IMPULSE_BINARY_DIR ${EDGE_IMPULSE_DIR}/src/edge_impulse_project-build)
set(EDGE_IMPULSE_STAMP_DIR ${EDGE_IMPULSE_DIR}/src/edge_impulse_project-stamp)
set(EDGE_IMPULSE_LIBRARY ${EDGE_IMPULSE_BINARY_DIR}/libedge_impulse.a)

# Generate compile options file for Edge Impulse external build
file(GENERATE OUTPUT ${EDGE_IMPULSE_DIR}/compile_options.$<COMPILE_LANGUAGE>.cmake CONTENT
  "set(EI_$<COMPILE_LANGUAGE>_COMPILE_OPTIONS \"$<TARGET_PROPERTY:zephyr_interface,INTERFACE_COMPILE_OPTIONS>\")"
)

###############################################################################
# Configuration
###############################################################################

# Collect all Edge Impulse header files for dependency tracking
file(GLOB_RECURSE edge_impulse_all_headers "${EDGE_IMPULSE_SOURCE_DIR}/*.h")

# Enable C linkage for Edge Impulse library (allows calling from C code)
target_compile_definitions(zephyr_interface INTERFACE
  EI_C_LINKAGE=1
  EIDSP_SIGNAL_C_FN_POINTER=1
)

###############################################################################
# External Project: build Edge Impulse library
###############################################################################

include(ExternalProject)
ExternalProject_Add(edge_impulse_project
  URL ${EI_URI_LIST}
  HTTP_HEADER "Accept: application/zip"
    ${EI_API_KEY_HEADER}
  DOWNLOAD_EXTRACT_TIMESTAMP True
  PREFIX ${EDGE_IMPULSE_DIR}
  SOURCE_DIR ${EDGE_IMPULSE_SOURCE_DIR}
  BINARY_DIR ${EDGE_IMPULSE_BINARY_DIR}
  STAMP_DIR ${EDGE_IMPULSE_STAMP_DIR}
  DOWNLOAD_NAME edge_impulse_src.zip
  BUILD_BYPRODUCTS ${EDGE_IMPULSE_LIBRARY}
    ${edge_impulse_all_headers}
  PATCH_COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_CURRENT_LIST_DIR}/CMakeLists.ei.template
    ${EDGE_IMPULSE_SOURCE_DIR}/CMakeLists.txt
  DEPENDS zephyr_interface
  CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_AR=${CMAKE_AR}
    -DCMAKE_RANLIB=${CMAKE_RANLIB}
    -DEI_COMPILE_DEFINITIONS=$<TARGET_PROPERTY:zephyr_interface,INTERFACE_COMPILE_DEFINITIONS>
    -DEI_INCLUDE_DIRECTORIES=$<TARGET_PROPERTY:zephyr_interface,INTERFACE_INCLUDE_DIRECTORIES>
    -DEI_SYSTEM_INCLUDE_DIRECTORIES=$<TARGET_PROPERTY:zephyr_interface,INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
    -DEI_LIBRARY_NAME=edge_impulse
  INSTALL_COMMAND ""
  BUILD_ALWAYS True
  USES_TERMINAL_BUILD True
)

##############################################################################
# Imported Library target
###############################################################################

# Create imported library target pointing to built libedge_impulse.a
add_library(edge_impulse_imported STATIC IMPORTED)
set_target_properties(edge_impulse_imported PROPERTIES
  IMPORTED_LOCATION ${EDGE_IMPULSE_LIBRARY}
)

add_dependencies(edge_impulse_project zephyr_generated_headers)

###############################################################################
# Force re-download on every build (optional)
###############################################################################

# This targets remove the `edge_impulse_project-download` stamp file created by
# ExternalProject, which causes the Edge impulse library to be fetched on each
# build invocation.
# Note: This also results in the `ALL` target to always be considered out-of-date.
if(CONFIG_EDGE_IMPULSE_DOWNLOAD_ALWAYS)
  if(${CMAKE_VERSION} VERSION_LESS "3.17")
    set(REMOVE_COMMAND remove)
  else()
    set(REMOVE_COMMAND rm)
  endif()

  add_custom_target(edge_impulse_project_download
    COMMAND ${CMAKE_COMMAND} -E ${REMOVE_COMMAND} -f
      ${EDGE_IMPULSE_STAMP_DIR}/edge_impulse_project-download
    DEPENDS edge_impulse_project
  )
endif()

##############################################################################
# Interface Library: application-facing API
###############################################################################

# Create interface library that bundles everything needed to use Edge Impulse
# from the application
zephyr_interface_library_named(edge_impulse)

# Provide include paths to Edge Impulse headers
target_include_directories(edge_impulse INTERFACE
  ${EDGE_IMPULSE_SOURCE_DIR}
)

# Link the actual library
target_link_libraries(edge_impulse INTERFACE edge_impulse_imported)

# Ensure proper build order
add_dependencies(edge_impulse
  zephyr_interface
  edge_impulse_project
  edge_impulse_project_download
)

# Link Edge Impulse library to application
target_link_libraries(app PRIVATE edge_impulse)
