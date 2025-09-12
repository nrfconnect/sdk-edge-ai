# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

#if (DEFINED BOARD)
    find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
    project(axon_app_base_inference)

    target_sources(app PRIVATE
            src/main.c
        )

    #
    # model name is specified through kConfig. However for automated builds the
    # environment variable AXON_MODEL_NAME can override kConfig.
    #
    if (DEFINED ENV{AXON_MODEL_NAME})
        message(STATUS "AXON_MODEL_NAME DEFINED $ENV{AXON_MODEL_NAME}, OVERRIDES CONFIG_AXON_MODEL_NAME ${CONFIG_AXON_MODEL_NAME}")
        zephyr_compile_definitions(
            AXON_MODEL_NAME=$ENV{AXON_MODEL_NAME}
        )
    else()
        message(STATUS "AXON_MODEL_NAME NOT DEFINED, USING CONFIG_AXON_MODEL_NAME ${CONFIG_AXON_MODEL_NAME}")
        zephyr_compile_definitions(
            AXON_MODEL_NAME=${CONFIG_AXON_MODEL_NAME}
        )
    endif()

    zephyr_include_directories(model/outputs)

    # Create symbolic link to compile_commands.json for IDE/tools
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
                    ${CMAKE_BINARY_DIR}/compile_commands.json
                    ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)

#else()
#    message(STATUS "Simulator build")
#endif()
