/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_COMMON_TYPES_H_
#define _NRF_EDGEAI_COMMON_TYPES_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 32-bit floating-point type definition.
 */
typedef float flt32_t;

/**
 * @brief 64-bit floating-point type definition.
 */
typedef double flt64_t;

/**
 * @brief Architecture 32-bit size_t type definition.
 */
typedef size_t size32_t;

/**
 * @brief Generic NRF EDGEAI operation status code
 */
typedef enum nrf_edgeai_err_e
{
    /** Operation is in progress */
    NRF_EDGEAI_ERR_INPROGRESS = 126,

    /** Operation successful */
    NRF_EDGEAI_ERR_SUCCESS = 0,

    /** The operation failed with an unspecified error */
    NRF_EDGEAI_ERR_UNSPECIFIED_ERROR = -1,

    /** The argument supplied to the operation was invalid */
    NRF_EDGEAI_ERR_INVALID_ARGUMENT = -2,

    /** The argument supplied to the operation was NULL */
    NRF_EDGEAI_ERR_NULL_ARGUMENT = -3,

    /** The requested operation was not available */
    NRF_EDGEAI_ERR_UNAVAILABLE = -4,

    /** The operation or service not supported */
    NRF_EDGEAI_ERR_NOT_SUPPORTED = -5,

    /** Wrong memory alignment */
    NRF_EDGEAI_ERR_WRONG_MEM_ALIGNMENT = -6,

    /** Incompatibilty error */
    NRF_EDGEAI_ERR_INCOMPATIBLE = -7,
} nrf_edgeai_err_t;

#ifdef __cplusplus
}
#endif

#endif /* _NRF_EDGEAI_COMMON_TYPES_H_ */