/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_PLATFORM_H_
#define _NRF_EDGEAI_PLATFORM_H_

#include "nrf_edgeai_debug.h"

#define __NRF_EDGEAI_STATIC_INLINE static inline

#if defined(__GNUC__)
#define __NRF_EDGEAI_WEAK               __attribute__((weak))
#define __NRF_EDGEAI_ALIGN(x)           __attribute__((aligned(x)))
#define __NRF_EDGEAI_ALIGNED            __attribute__((aligned))
#define __NRF_EDGEAI_PACKED             __attribute__((__packed__))
#define __NRF_EDGEAI_STATIC_FORCEINLINE __attribute__((always_inline)) static inline
#elif defined(__ICCARM__)
#define __NRF_EDGEAI_WEAK               __weak
#define __NRF_EDGEAI_PACKED             __packed
#define __NRF_EDGEAI_ALIGNED            __attribute__((aligned))
#define __NRF_EDGEAI_ALIGN(x)           __attribute__((aligned(x)))
#define __NRF_EDGEAI_STATIC_FORCEINLINE _Pragma("inline=forced") __NRF_EDGEAI_STATIC_INLINE
#elif defined(__ARMCC_VERSION) && (__ARMCC_VERSION >= 6010050) /* ARM Compiler V6 */
#define __NRF_EDGEAI_WEAK               __attribute__((weak))
#define __NRF_EDGEAI_ALIGN(x)           __attribute__((aligned(x)))
#define __NRF_EDGEAI_ALIGNED            __attribute__((aligned))
#define __NRF_EDGEAI_PACKED             __attribute__((packed))
#define __NRF_EDGEAI_STATIC_FORCEINLINE __NRF_EDGEAI_STATIC_INLINE
#else
#define __NRF_EDGEAI_WEAK               __attribute__((weak))
#define __NRF_EDGEAI_ALIGN(x)           __attribute__((aligned(x)))
#define __NRF_EDGEAI_ALIGNED            __attribute__((aligned))
#define __NRF_EDGEAI_PACKED             __attribute__((__packed__))
#define __NRF_EDGEAI_STATIC_FORCEINLINE __NRF_EDGEAI_STATIC_INLINE
#endif

#define NRF_EDGEAI_UNUSED(x) ((void)(x))
#define NRF_EDGEAI_VOID_VALUE

#ifndef __UNUSED_ARG
#define __UNUSED_ARG NRF_EDGEAI_UNUSED
#endif // __UNUSED_ARG

#ifndef __VOID_VALUE
#define __VOID_VALUE NRF_EDGEAI_VOID_VALUE
#endif // __VOID_VALUE

#endif /* _NRF_EDGEAI_PLATFORM_H_ */