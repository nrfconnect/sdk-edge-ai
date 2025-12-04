/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_NN_PLATFORM_H_
#define _NRF_NN_PLATFORM_H_

#include <nrf_edgeai/nrf_edgeai_platform.h>

#include "nrf_nn_platform_debug.h"
#include "nrf_nn_platform_types.h"

#include <math.h>

#define __NRF_NN_STATIC_INLINE      __NRF_EDGEAI_STATIC_INLINE
#define __NRF_NN_WEAK               __NRF_EDGEAI_WEAK
#define __NRF_NN_ALIGN              __NRF_EDGEAI_ALIGN
#define __NRF_NN_ALIGNED            __NRF_EDGEAI_ALIGNED
#define __NRF_NN_PACKED             __NRF_EDGEAI_PACKED
#define __NRF_NN_STATIC_FORCEINLINE __NRF_EDGEAI_STATIC_FORCEINLINE

#define NRF_NN_UNUSED     NRF_EDGEAI_UNUSED
#define NRF_NN_VOID_VALUE NRF_EDGEAI_VOID_VALUE

#endif /* _NRF_NN_PLATFORM_H_ */