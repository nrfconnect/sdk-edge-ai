# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

if(NOT DEFINED MODEL_IMAGE_O OR NOT DEFINED MODEL_IMAGE_ELF
   OR NOT DEFINED MODEL_IMAGE_BIN OR NOT DEFINED SYMS_HEADER
   OR NOT DEFINED MODEL_PARTITION_ADDR OR NOT DEFINED LINKER_SCRIPT
   OR NOT DEFINED INCLUDE_DIR_PARTITION OR NOT DEFINED INCLUDE_DIR_EDGE_AI
   OR NOT DEFINED CMAKE_C_COMPILER OR NOT DEFINED CMAKE_OBJCOPY)
  message(FATAL_ERROR "build_model_image.cmake: missing required variables")
endif()

if(NOT DEFINED MODEL_IMAGE_C AND NOT DEFINED MODEL_IMAGE_STUB_C)
  message(FATAL_ERROR "build_model_image.cmake: MODEL_IMAGE_C or MODEL_IMAGE_STUB_C is required")
endif()

if(DEFINED MODEL_IMAGE_STUB_C)
  if(NOT DEFINED MODEL_FIXUPS_HEADER OR NOT DEFINED MODEL_HEADER_DIR)
    message(FATAL_ERROR "build_model_image.cmake: stub build requires MODEL_FIXUPS_HEADER and MODEL_HEADER_DIR")
  endif()
  set(model_image_source ${MODEL_IMAGE_STUB_C})
  set(model_image_compile_flags
    -I${MODEL_HEADER_DIR}
    -include ${MODEL_FIXUPS_HEADER}
  )
  if(DEFINED NRF_AXON_INTERLAYER_BUFFER_SIZE)
    list(APPEND model_image_compile_flags
      -DNRF_AXON_INTERLAYER_BUFFER_SIZE=${NRF_AXON_INTERLAYER_BUFFER_SIZE})
  endif()
else()
  set(model_image_source ${MODEL_IMAGE_C})
  set(model_image_compile_flags "")
endif()

execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -c ${model_image_source}
    -o ${MODEL_IMAGE_O}
    -I${INCLUDE_DIR_PARTITION}
    -I${INCLUDE_DIR_EDGE_AI}
    -include ${SYMS_HEADER}
    -DNRF_AXON_MODEL_PARTITION_ADDR=${MODEL_PARTITION_ADDR}
    ${model_image_compile_flags}
  COMMAND_ERROR_IS_FATAL ANY
)

set(linker_script_args -T ${LINKER_SCRIPT})
if(DEFINED MODEL_IMAGE_STUB_C AND DEFINED SYMS_LINKER_SCRIPT)
  list(APPEND linker_script_args -T ${SYMS_LINKER_SCRIPT})
endif()

execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -nostdlib
    -nostartfiles
    -Wl,--gc-sections
    -Wl,--defsym=NRF_AXON_MODEL_PARTITION_ADDR=${MODEL_PARTITION_ADDR}
    ${linker_script_args}
    -o ${MODEL_IMAGE_ELF}
    ${MODEL_IMAGE_O}
  COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
  COMMAND ${CMAKE_OBJCOPY}
    -O binary
    -j .model_image
    ${MODEL_IMAGE_ELF}
    ${MODEL_IMAGE_BIN}
  COMMAND_ERROR_IS_FATAL ANY
)
