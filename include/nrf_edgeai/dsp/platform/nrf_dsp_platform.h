/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_DSP_PLATFORM_H_
#define _NRF_DSP_PLATFORM_H_

#include <nrf_edgeai/nrf_edgeai_platform.h>

#include "nrf_dsp_platform_debug.h"
#include "nrf_dsp_platform_types.h"

#include <math.h>

#define __NRF_DSP_STATIC_INLINE      __NRF_EDGEAI_STATIC_INLINE
#define __NRF_DSP_WEAK               __NRF_EDGEAI_WEAK
#define __NRF_DSP_ALIGN              __NRF_EDGEAI_ALIGN
#define __NRF_DSP_ALIGNED            __NRF_EDGEAI_ALIGNED
#define __NRF_DSP_PACKED             __NRF_EDGEAI_PACKED
#define __NRF_DSP_STATIC_FORCEINLINE __NRF_EDGEAI_STATIC_FORCEINLINE

#define NRF_DSP_UNUSED     NRF_EDGEAI_UNUSED
#define NRF_DSP_VOID_VALUE NRF_EDGEAI_VOID_VALUE

#endif /* _NRF_DSP_PLATFORM_H_ */