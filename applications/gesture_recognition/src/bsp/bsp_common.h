/**
 *
 * @defgroup bsp_common Common
 * @{
 * @ingroup bsp
 *
 * @brief BSP common useful utils, types and macro.
 *
 */
#ifndef __BSP_COMMON_H__
#define __BSP_COMMON_H__

#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


/**
 * @brief Macro for performing rounded integer division (as opposed to truncating the result).
 *
 * @param[in]   A   Numerator.
 * @param[in]   B   Denominator.
 *
 * @return      Rounded (integer) result of dividing A by B.
 */
#ifndef ROUNDED_DIV
#define ROUNDED_DIV(A, B) (((A) + ((B) / 2)) / (B))
#endif

/**
 * @brief Macro for counting items in an object.
 *
 * @param[in]   x   An object for which the counting will be made.
 *
 * @return      A number of items in an object.
 */
#ifndef COUNT_OF
#define COUNT_OF(x) ((sizeof(x)/sizeof(0[x])) / ((size_t)(!(sizeof(x) % sizeof(0[x])))))
#endif

/**
 * @brief Macro for getting offset for a field in the provided type.
 *
 * @param[in]   type   Provided type.
 * @param[in]   field  Field in the provided type.
 *
 * @return         Offset of a field in the provided type.
 */
#ifndef OFFSET_OF
#define OFFSET_OF(type, field)    ((unsigned long) &(((type *) 0)->field))
#endif

/**
 * @brief Macro for obtaining the minimum value of two.
 *
 * @param[in]   A  First value.
 * @param[in]   B  Second value.
 *
 * @return     Minimum value of two set values.
 */
#ifndef MIN
#define MIN(A, B) (( (A) < (B) ) ? (A) : (B))
#endif

/**
 * @brief Macro for obtaining the maximum value of two.
 *
 * @param[in]   A  First value.
 * @param[in]   B  Second value.
 *
 * @return     Maximum value of two set values.
 */
#ifndef MAX
#define MAX(A, B) (( (A) > (B) ) ? (A) : (B))
#endif

/**
 * @brief Macro for obtaining the constraint value between min and max values.
 *
 * @param[in]   val  input value.
 * @param[in]   vmin lower border.
 * @param[in]   vmax higher border.
 *
 * @return     Constraint value between min and max.
 */
#ifndef CONSTRAIN
#define CONSTRAIN(val, vmin, vmax) ((val)>(vmax)?(vmax):(val)<(vmin)?(vmin):(val))
#endif

/**
 * @brief Macro for unused argument.
 *
 * @param[in]   x  Unused argument.
 *
 * @return     None.
 */
#ifndef UNUSED
#define UNUSED(x) ((void)(x))
#endif

/** @brief Void value using for returns or argument for void-argument functions. */
#define VOID_VALUE

/**
 * @brief Macro for checking argument for NULL.
 *        Macro will call return with the status BSP_STATUS_NULL_ARGUMENT.
 *
 * @param[in]   x  Argument to be checked.
 *
 */
#define BSP_NULL_CHECK(x)                               \
    do                                                  \
    {                                                   \
        if ( (x) == NULL )                              \
        {                                               \
            return BSP_STATUS_NULL_ARGUMENT;            \
        }                                               \
    } while(0)

/**
 * @brief Macro for verifying that the provided status is BSP_STATUS_SUCCESS. It will cause the exterior
 *        function to return an error code if it is not @ref BSP_STATUS_SUCCESS.
 *
 * @param[in] status     Status to check vs BSP_STATUS_SUCCESS.
 */
#define BSP_VERIFY_SUCCESS(status) \
do                                                      \
{                                                       \
    if ((status) != BSP_STATUS_SUCCESS)                 \
    {                                                   \
        return (status);                                \
    }                                                   \
} while(0)

/**
 * @brief Macro for verifying that the provided argumets is valid. It will cause the exterior
 *        function to return an error code if it is not @ref BSP_STATUS_INVALID_ARGUMENT.
 *
 * @param[in] is_valid     boolean comparison on the validity of the argument.
 */
#define BSP_VERIFY_VALID_ARG(is_valid)                  \
do                                                      \
{                                                       \
    if (!(is_valid))                                    \
    {                                                   \
        return BSP_STATUS_INVALID_ARGUMENT;             \
    }                                                   \
} while(0)

/**
 * @brief Macro for verifying any boolean condition and returning status if condition failed
 *
 * @param[in] err_cond    boolean condition to be checked.
 * @param[in] err         Return status if condition failed.
 */
#define BSP_RETURN_IF(err_cond, err) __RETURN_CONDITIONAL(err_cond, err)

/**
 * @brief Return if expr == true.
 *
 * @param[in]   expr    Expression for validating.
 * @param[in]   ret_val Returning value.
 */
#ifndef __RETURN_CONDITIONAL
#    define __RETURN_CONDITIONAL(expr, ret_val) \
        do                                      \
        {                                       \
            if ((expr) == true)                 \
            {                                   \
                return ret_val;                 \
            }                                   \
        }                                       \
        while (0)
#endif

/** Generic callback type */
typedef void(*bsp_generic_cb_t)(void);

/** Base interrupt request handler type */
typedef bsp_generic_cb_t bsp_irq_handler_t;

/**
 * @brief Base async data ready callback type
 *
 * @param[in] data        A pointer to the data buffer that was passed to the async function
 * @param[in] data_size   Size of data that was passed to the async function, in bytes
 */
typedef void(*bsp_async_drdy_cb_t)(void* data, uint32_t data_size);

/**
 * Generic bsp operation status code
 *
 * This enumeration is used by various bsp subsystems to provide
 * information on their status. It may also be used by functions as a
 * return code.
 */
typedef enum bsp_status_e
{
    /** Operation successful */
    BSP_STATUS_SUCCESS,

    /** The operation failed because the module is already in the
     * requested mode */
    BSP_STATUS_ALREADY_IN_MODE,

    /** There was an error communicating with hardware */
    BSP_STATUS_HARDWARE_ERROR,

    /** The operation failed with an unspecified error */
    BSP_STATUS_UNSPECIFIED_ERROR,

    /** The argument supplied to the operation was invalid */
    BSP_STATUS_INVALID_ARGUMENT,

    /** The argument supplied to the operation was NULL */
    BSP_STATUS_NULL_ARGUMENT,

    /** The operation failed because the module was busy */
    BSP_STATUS_BUSY,

    /** The requested operation was not available */
    BSP_STATUS_UNAVAILABLE,

    /** The operation or service not supported */
    BSP_STATUS_NOT_SUPPORTED,

    /** The requested operation timeout */
    BSP_STATUS_TIMEOUT,
} bsp_status_t;

#ifdef __cplusplus
}
#endif /* __cplusplus */


#endif /* __BSP_COMMON_H__ */

/**
 * @}
 */
