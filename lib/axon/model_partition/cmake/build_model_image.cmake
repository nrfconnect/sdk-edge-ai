# SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
#
# Second-stage compile/link script for the Axon model partition image.
# Invoked from nrf_axon_model_partition.cmake after zephyr.elf exists.

if(NOT DEFINED MODEL_IMAGE_STUB_C OR NOT DEFINED MODEL_FIXUPS_HEADER
   OR NOT DEFINED MODEL_HEADER_DIR OR NOT DEFINED MODEL_IMAGE_O
   OR NOT DEFINED MODEL_IMAGE_ELF OR NOT DEFINED MODEL_IMAGE_BIN
   OR NOT DEFINED SYMS_HEADER OR NOT DEFINED SYMS_LINKER_SCRIPT
   OR NOT DEFINED MODEL_PARTITION_ADDR OR NOT DEFINED LINKER_SCRIPT
   OR NOT DEFINED INCLUDE_DIR_PARTITION OR NOT DEFINED INCLUDE_DIR_EDGE_AI
   OR NOT DEFINED CMAKE_C_COMPILER OR NOT DEFINED CMAKE_OBJCOPY
   OR NOT DEFINED CMAKE_NM OR NOT DEFINED VALIDATE_LAYOUT_SCRIPT
   OR NOT DEFINED PYTHON_EXECUTABLE OR NOT DEFINED PARTITION_DEFS_HEADER)
  message(FATAL_ERROR "build_model_image.cmake: missing required variables")
endif()

set(model_image_compile_flags
  -I${MODEL_HEADER_DIR}
  -include ${MODEL_FIXUPS_HEADER}
)

# Match the application's interlayer buffer size so offsetof checks in the model
# header compile identically in both link passes.
if(DEFINED NRF_AXON_INTERLAYER_BUFFER_SIZE)
  list(APPEND model_image_compile_flags
    -DNRF_AXON_INTERLAYER_BUFFER_SIZE=${NRF_AXON_INTERLAYER_BUFFER_SIZE})
endif()

# Compile stub once: fixups header pulls in model rodata; syms header is empty but
# the companion linker script supplies PROVIDE() symbols for app-owned pointers.
execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -c ${MODEL_IMAGE_STUB_C}
    -o ${MODEL_IMAGE_O}
    -I${INCLUDE_DIR_PARTITION}
    -I${INCLUDE_DIR_EDGE_AI}
    -include ${SYMS_HEADER}
    -DNRF_AXON_MODEL_PARTITION_ADDR=${MODEL_PARTITION_ADDR}
    ${model_image_compile_flags}
  COMMAND_ERROR_IS_FATAL ANY
)

# Link at partition base. model_image.ld lays out header + rodata; syms linker
# script patches absolute addresses for buffers living in the application.
execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -nostdlib
    -nostartfiles
    -Wl,--gc-sections
    -Wl,--defsym=NRF_AXON_MODEL_PARTITION_ADDR=${MODEL_PARTITION_ADDR}
    -T ${LINKER_SCRIPT}
    -T ${SYMS_LINKER_SCRIPT}
    -o ${MODEL_IMAGE_ELF}
    ${MODEL_IMAGE_O}
  COMMAND_ERROR_IS_FATAL ANY
)

# Emit raw bytes for the .model_image output section only.
execute_process(
  COMMAND ${CMAKE_OBJCOPY}
    -O binary
    -j .model_image
    ${MODEL_IMAGE_ELF}
    ${MODEL_IMAGE_BIN}
  COMMAND_ERROR_IS_FATAL ANY
)

# Fail the build if header fields disagree with linker anchors or model symbol.
execute_process(
  COMMAND ${PYTHON_EXECUTABLE}
    ${VALIDATE_LAYOUT_SCRIPT}
    --nm ${CMAKE_NM}
    --elf ${MODEL_IMAGE_ELF}
    --bin ${MODEL_IMAGE_BIN}
    --partition-addr ${MODEL_PARTITION_ADDR}
    --fixups-header ${MODEL_FIXUPS_HEADER}
    --defs-header ${PARTITION_DEFS_HEADER}
  COMMAND_ERROR_IS_FATAL ANY
)
