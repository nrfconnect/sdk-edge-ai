/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */
/**
 *
 * @defgroup nrf_edgeai_debug Debug Utils & Macros
 * @{
 *
 * @brief
 *
 */

#ifndef _NRF_EDGEAI_DEBUG_H_
#define _NRF_EDGEAI_DEBUG_H_

#include <nrf_edgeai/nrf_edgeai_macro.h>

/**
 * @brief Logging macro, can be enabled by setting NRF_EDGEAI_LOG_ENABLE to 1
 * 
 */
#ifndef NRF_EDGEAI_LOG_ENABLE
#define NRF_EDGEAI_LOG_ENABLE 0
#endif

#if NRF_EDGEAI_LOG_ENABLE == 1
#include <stdio.h>
#define NRF_EDGEAI_LOG(...) printf(__VA_ARGS__)
#else
#define NRF_EDGEAI_LOG(...)
#endif  // NRF_DSP_LOG_ENABLE

/**
 * @brief Assertion macro, can be enabled by setting NRF_EDGEAI_ASSERT_ENABLE to 1
 * 
 */
#ifndef NRF_EDGEAI_ASSERT_ENABLE
#define NRF_EDGEAI_ASSERT_ENABLE 0
#endif

#if NRF_EDGEAI_ASSERT_ENABLE == 1
#include <assert.h>
#define nrf_edgeai_assert(x) assert(x)
#else
#define nrf_edgeai_assert(x) ((void)(x))
#endif  // NRF_EDGEAI_ASSERT_ENABLE

#endif /* _NRF_EDGEAI_DEBUG_H_ */

/**
 * @}
 */
