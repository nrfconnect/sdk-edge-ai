/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @defgroup hw_modules HW modules for sensor based application
 * @ingroup app_sbm
 */

/**
 *
 * @defgroup common Common
 * @{
 * @ingroup hw_modules
 *
 * @brief Common useful utils, types and macro.
 *
 */
#ifndef __COMMON_H__
#define __COMMON_H__

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Macro for checking argument for NULL.
 *        Macro will call return with the error code -EFAULT.
 *
 * @param[in]   x  Argument to be checked.
 *
 */
#define NULL_CHECK(x)					\
	do {						\
		if ((x) == NULL) {			\
			return -EFAULT;			\
		}					\
	} while (0)

/**
 * @brief Macro for verifying that the provided argumets is valid. It will cause the exterior
 *        function to return the error code -EINVAL if it is not.
 *
 * @param[in] is_valid     boolean comparison on the validity of the argument.
 */
#define VERIFY_VALID_ARG(is_valid)			\
do {							\
	if (!(is_valid)) {				\
		return -EINVAL;				\
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

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __COMMON_H__ */

/**
 * @}
 */
