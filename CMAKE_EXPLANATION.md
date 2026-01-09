# CMake Files in the Edge AI SDK

This document explains what happens in the CMake files within the nRF Edge AI SDK repository.

**Note:** This repository uses "nrf_edgeai" naming convention, not "edge_impulse". If you're looking for Edge Impulse integration, this SDK provides a framework for running machine learning models on Nordic devices, potentially including models from various sources.

## Overview of CMake Structure

The Edge AI SDK uses CMake to build machine learning models and integrate them with the Zephyr RTOS on Nordic Semiconductor devices. The build system is organized into several layers:

1. **Top-level integration** (`zephyr/CMakeLists.txt`)
2. **Library configuration** (`lib/CMakeLists.txt` and `lib/nrf_edgeai/CMakeLists.txt`)
3. **Architecture-specific libraries** (`lib/nrf_edgeai/cortex-m33/` and `lib/nrf_edgeai/cortex-m4/`)
4. **Sample applications** (`samples/nrf_edgeai/*/CMakeLists.txt`)
5. **Generated model code** (`samples/nrf_edgeai/*/src/nrf_edgeai_generated/CMakeLists.txt`)

---

## 1. Top-Level Integration: `zephyr/CMakeLists.txt`

**Location:** `/home/runner/work/sdk-edge-ai/sdk-edge-ai/zephyr/CMakeLists.txt`

```cmake
zephyr_include_directories(${ZEPHYR_CURRENT_MODULE_DIR}/include)
add_subdirectory(${ZEPHYR_CURRENT_MODULE_DIR}/lib lib)
```

**What it does:**
- Integrates the Edge AI SDK as a Zephyr module
- Adds the SDK's include directories to the build path so header files are accessible
- Includes the library subdirectory to build the Edge AI runtime libraries

**Key variables:**
- `ZEPHYR_CURRENT_MODULE_DIR`: Points to the SDK's root directory

---

## 2. Library Entry Point: `lib/CMakeLists.txt`

**Location:** `/home/runner/work/sdk-edge-ai/sdk-edge-ai/lib/CMakeLists.txt`

```cmake
if (CONFIG_NRF_EDGEAI)
    add_subdirectory(nrf_edgeai)
endif(CONFIG_NRF_EDGEAI)
```

**What it does:**
- Conditionally includes the nrf_edgeai library based on Kconfig settings
- Only builds Edge AI support if `CONFIG_NRF_EDGEAI` is enabled in the project configuration

**Dependencies:**
- Requires `CONFIG_NRF_EDGEAI=y` in prj.conf or Kconfig

---

## 3. CPU Architecture Detection: `lib/nrf_edgeai/CMakeLists.txt`

**Location:** `/home/runner/work/sdk-edge-ai/sdk-edge-ai/lib/nrf_edgeai/CMakeLists.txt`

```cmake
if(CONFIG_CPU_CORTEX_M4)
    message(STATUS "Building for Cortex-M4 CPU")
    add_subdirectory(cortex-m4)
elseif(CONFIG_CPU_CORTEX_M33)
    message(STATUS "Building for Cortex-M33 CPU")
    add_subdirectory(cortex-m33)
else()
    message(FATAL_ERROR "Edge AI is not supported on the selected CPU architecture.")
endif()
```

**What it does:**
- Detects the target CPU architecture (Cortex-M4 or Cortex-M33)
- Selects the appropriate precompiled library for the target architecture
- Stops the build with an error if an unsupported CPU is detected

**Supported architectures:**
- ARM Cortex-M4 (e.g., nRF52 series)
- ARM Cortex-M33 (e.g., nRF5340, nRF9160)

---

## 4. Architecture-Specific Libraries

### Cortex-M33: `lib/nrf_edgeai/cortex-m33/CMakeLists.txt`

**Location:** `/home/runner/work/sdk-edge-ai/sdk-edge-ai/lib/nrf_edgeai/cortex-m33/CMakeLists.txt`

```cmake
set(SID_LIB_DIR ${CMAKE_CURRENT_LIST_DIR})

zephyr_library_link_libraries(
    ${SID_LIB_DIR}/libnrf_edgeai_cortex-m33.a 
)
```

**What it does:**
- Links the precompiled Edge AI static library for Cortex-M33
- The library contains optimized DSP functions, neural network inference engine, and runtime support
- Uses `zephyr_library_link_libraries()` to integrate with Zephyr's build system

**Library file:** `libnrf_edgeai_cortex-m33.a` (precompiled static library)

### Cortex-M4: `lib/nrf_edgeai/cortex-m4/CMakeLists.txt`

**Location:** `/home/runner/work/sdk-edge-ai/sdk-edge-ai/lib/nrf_edgeai/cortex-m4/CMakeLists.txt`

```cmake
set(SID_LIB_DIR ${CMAKE_CURRENT_LIST_DIR})

zephyr_library_link_libraries(
    ${SID_LIB_DIR}/libnrf_edgeai_cortex-m4.a 
)
```

**What it does:**
- Same as Cortex-M33 version but for Cortex-M4 architecture
- Links the precompiled Edge AI static library optimized for Cortex-M4

**Library file:** `libnrf_edgeai_cortex-m4.a` (precompiled static library)

---

## 5. Sample Applications

### Example: Classification Sample

**Location:** `/home/runner/work/sdk-edge-ai/sdk-edge-ai/samples/nrf_edgeai/classification/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.20.0)

# Zephyr CMake project
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(nrf_edgeai_classification_sample)

# Source files
target_include_directories(app PRIVATE include)
target_sources(app PRIVATE
    src/main.c
    src/nrf_edgeai_generated/nrf_edgeai_user_model.c
)
```

**What it does:**
- Defines a complete Zephyr application for running a classification ML model
- Finds and configures the Zephyr build system
- Adds application-specific source files:
  - `main.c`: Application logic
  - `nrf_edgeai_user_model.c`: Generated model code
- Sets up include directories for application headers

**Similar files exist for:**
- `samples/nrf_edgeai/anomaly/CMakeLists.txt` - Anomaly detection sample
- `samples/nrf_edgeai/regression/CMakeLists.txt` - Regression model sample

---

## 6. Generated Model Code: `nrf_edgeai_generated/CMakeLists.txt`

**Location:** All three sample applications have identical files at:
- `/home/runner/work/sdk-edge-ai/sdk-edge-ai/samples/nrf_edgeai/classification/src/nrf_edgeai_generated/CMakeLists.txt`
- `/home/runner/work/sdk-edge-ai/sdk-edge-ai/samples/nrf_edgeai/anomaly/src/nrf_edgeai_generated/CMakeLists.txt`
- `/home/runner/work/sdk-edge-ai/sdk-edge-ai/samples/nrf_edgeai/regression/src/nrf_edgeai_generated/CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.13)

project(NRF_EdgeAI_User_Model
   VERSION 1.0
   LANGUAGES
   C
)

add_library(${PROJECT_NAME}
   nrf_edgeai_user_model.c
)

add_library(NRF::EdgeAI::UserModel ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME}
   PRIVATE
       NRF::EdgeAI
)
```

**What it does:**
- Creates a standalone library for user-generated ML model code
- Compiles `nrf_edgeai_user_model.c` which contains:
  - Model architecture definition
  - Neural network weights and parameters
  - Input/output specifications
  - Feature extraction configuration
- Creates an alias `NRF::EdgeAI::UserModel` for easy linking
- Links against the main Edge AI runtime library (`NRF::EdgeAI`)

**Generated files in this directory:**
- `nrf_edgeai_user_model.c` - Main model implementation
- `nrf_edgeai_user_model.h` - Model interface
- `nrf_edgeai_user_types.h` - Model-specific type definitions

These files are typically generated by the Nordic Edge AI Lab or exported from ML training tools.

---

## Build Flow Summary

1. **Configuration Phase:**
   - User enables `CONFIG_NRF_EDGEAI` in Kconfig
   - CMake detects target CPU architecture (M4 or M33)

2. **Library Linking:**
   - Appropriate precompiled Edge AI library is linked
   - Zephyr module integration makes headers available

3. **Model Compilation:**
   - Generated model code is compiled into a library
   - Model library links against Edge AI runtime

4. **Application Build:**
   - Application code links against both the model library and Edge AI runtime
   - Final binary contains ML model ready to run on device

---

## Key Concepts

### Precompiled Libraries
The Edge AI SDK uses precompiled static libraries (`libnrf_edgeai_cortex-m*.a`) rather than source code. This:
- Protects proprietary algorithms
- Reduces build time
- Provides optimized implementations for each architecture

### Generated Model Code
The `nrf_edgeai_generated` directories contain code that is generated by external tools (Nordic Edge AI Lab). Users replace these files with their own trained models.

### Zephyr Integration
All CMake files integrate with Zephyr's build system using:
- `find_package(Zephyr)` - Locates Zephyr
- `zephyr_library_link_libraries()` - Links libraries
- `zephyr_include_directories()` - Adds include paths
- `target_sources(app ...)` - Adds sources to Zephyr app

---

## How to Use These Files

### For Application Developers:
1. Enable Edge AI in your `prj.conf`: `CONFIG_NRF_EDGEAI=y`
2. Generate your model code using Nordic Edge AI Lab
3. Replace the contents of `nrf_edgeai_generated/` with your generated files
4. Build your application - CMake handles the rest

### For SDK Maintainers:
- The architecture detection ensures correct library selection
- Update library versions by replacing `.a` files in `cortex-m*/` directories
- The structure supports easy addition of new architectures

---

## Additional Resources

- **Edge AI Documentation:** https://docs.nordicsemi.com/bundle/addon-edge-ai_v1.0.0-rc2/page/index.html
- **Nordic Edge AI Lab:** https://docs.nordicsemi.com/bundle/edge-ai-lab/page/doc/overview_of_nordic_edge_ai_lab.html
- **Zephyr CMake Documentation:** https://docs.zephyrproject.org/latest/build/cmake/index.html

---

## Common Questions

**Q: Where is the "edge_impulse" folder?**  
A: This repository uses "nrf_edgeai" naming. If you're looking for Edge Impulse integration, you would need to generate models from Edge Impulse Studio and adapt them to the nrf_edgeai format, or use a different integration method.

**Q: Can I see the source code of the Edge AI library?**  
A: No, the core runtime is provided as precompiled static libraries (`.a` files). Only the API headers are available in the `include/` directory.

**Q: How do I add support for a new CPU architecture?**  
A: You would need:
1. A precompiled library for that architecture
2. A new subdirectory in `lib/nrf_edgeai/cortex-*/`
3. Update to `lib/nrf_edgeai/CMakeLists.txt` to detect and include the new architecture

**Q: Why are there three identical CMakeLists.txt in nrf_edgeai_generated folders?**  
A: Each sample (classification, anomaly, regression) is independent. While the CMake files are currently identical, they could diverge if different samples need different build configurations in the future.
