/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_DSP_PLATFORM_TYPES_H_
#define _NRF_DSP_PLATFORM_TYPES_H_

#ifndef NRF_DSP_CUSTOM_PLATFORM_TYPES
#include <nrf_edgeai/nrf_edgeai_ctypes.h>
#else
// User-defined platform types
#include <nrf_dsp_custom_platform_types.h>
#endif  // NRF_DSP_CUSTOM_PLATFORM_TYPES

// Header for defining platform-specific types for DSP library:
// typedef float flt32_t;
// typedef double flt64_t;
// typedef size_t size32_t;

#endif /* _NRF_DSP_PLATFORM_TYPES_H_ */