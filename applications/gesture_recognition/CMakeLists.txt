# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_thingy53)

set(CMAKE_BUILD_TYPE Debug)

file(GLOB_RECURSE APP_SOURCE_FILES
        "${CMAKE_CURRENT_LIST_DIR}/src/**")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/bsp) 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_lib/nrf_edgeai/include)        
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/nrf_edgeai_lib)      

target_sources(app PRIVATE ${APP_SOURCE_FILES})

zephyr_library_include_directories(${ZEPHYR_BASE}/samples/bluetooth)
zephyr_link_libraries(${CMAKE_CURRENT_LIST_DIR}/src/nrf_edgeai_lib/nrf_edgeai/lib/libnrf_edgeai_cortex-m33.a)