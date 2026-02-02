/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 *
 * @defgroup common Common
 * @{
 * @ingroup bsp
 *
 * @brief BSP common useful utils, types and macro.
 *
 */
#ifndef __COMMON_H__
#define __COMMON_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Macro for checking argument for NULL.
 *        Macro will call return with the status STATUS_NULL_ARGUMENT.
 *
 * @param[in]   x  Argument to be checked.
 *
 */
#define NULL_CHECK(x)					\
	do {						\
		if ((x) == NULL) {			\
			return STATUS_NULL_ARGUMENT;	\
		}					\
	} while (0)

/**
 * @brief Macro for verifying that the provided argumets is valid. It will cause the exterior
 *        function to return an error code if it is not @ref STATUS_INVALID_ARGUMENT.
 *
 * @param[in] is_valid     boolean comparison on the validity of the argument.
 */
#define VERIFY_VALID_ARG(is_valid)			\
do {							\
	if (!(is_valid)) {				\
		return STATUS_INVALID_ARGUMENT;		\
	}						\
} while (0)

/**
 * @brief Macro for verifying any boolean condition and returning status if condition failed
 *
 * @param[in] err_cond    boolean condition to be checked.
 * @param[in] err         Return status if condition failed.
 */
#define HW_RETURN_IF(err_cond, err) __RETURN_CONDITIONAL(err_cond, err)

/**
 * @brief Return if expr == true.
 *
 * @param[in]   expr    Expression for validating.
 * @param[in]   ret_val Returning value.
 */
#ifndef __RETURN_CONDITIONAL
#    define __RETURN_CONDITIONAL(expr, ret_val)		\
		do {					\
			if ((expr) == true) {		\
				return ret_val;		\
			}				\
		}					\
		while (0)
#endif

/** Generic callback type */
typedef void(*generic_cb_t)(void);

/** Base interrupt request handler type */
typedef generic_cb_t irq_handler_t;

/**
 * @brief Base async data ready callback type
 *
 * @param[in] data        A pointer to the data buffer that was passed to the async function
 * @param[in] data_size   Size of data that was passed to the async function, in bytes
 */
typedef void(*async_drdy_cb_t)(void *data, uint32_t data_size);

/**
 * Generic bsp operation status code
 *
 * This enumeration is used by various bsp subsystems to provide
 * information on their status. It may also be used by functions as a
 * return code.
 */
typedef enum status_e {
	/** Operation successful */
	STATUS_SUCCESS,

	/** The operation failed because the module is already in the requested mode */
	STATUS_ALREADY_IN_MODE,

	/** There was an error communicating with hardware */
	STATUS_HARDWARE_ERROR,

	/** The operation failed with an unspecified error */
	STATUS_UNSPECIFIED_ERROR,

	/** The argument supplied to the operation was invalid */
	STATUS_INVALID_ARGUMENT,

	/** The argument supplied to the operation was NULL */
	STATUS_NULL_ARGUMENT,

	/** The operation failed because the module was busy */
	STATUS_BUSY,

	/** The requested operation was not available */
	STATUS_UNAVAILABLE,

	/** The operation or service not supported */
	STATUS_NOT_SUPPORTED,

	/** The requested operation timeout */
	STATUS_TIMEOUT,
} status_t;

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __COMMON_H__ */

/**
 * @}
 */
