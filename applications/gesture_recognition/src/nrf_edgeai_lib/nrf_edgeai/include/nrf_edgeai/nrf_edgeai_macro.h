/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _NRF_EDGEAI_MACRO_H_
#define _NRF_EDGEAI_MACRO_H_

/**
 * @brief Macro for verifying that the provided condition is true. It will cause the exterior
 *        function to return an error value if it is not.
 *
 * @param[in] is_true     Boolean condition to check.
 * @param[in] ret_val     The value to be returned if the checks fails
 */
#ifndef RETURN_IF
#define RETURN_IF(is_true, ret_val)        \
    do                                     \
    {                                      \
        if ((is_true)) { return ret_val; } \
    }                                      \
    while (0)
#endif  // RETURN_IF

/**
 * @brief Creates a bit mask from one unit among the remaining zeros (for example 00..00000010).
 *
 * @param pos Position of one-bit.
 *
 * @return Bitmask.
 */
#ifndef BIT_ONE
#define BIT_ONE(pos) (1UL << (pos))
#endif

/**
 * @brief Creates a bit mask from one zero among the remaining ones (for example 11..11101111).
 *
 * @param pos Position of one-bit.
 *
 * @return Bitmask.
 */
#ifndef BIT_ZERO
#define BIT_ZERO(pos) (~BIT_ONE(pos))
#endif

/**
 * @brief Checks that bit is set.
 *
 * @param value Value that for test bit.
 * @param pos   Position number of the bit.
 *
 * @return True if this bit is set, else in otherwise.
 */
#ifndef BIT_IS_SET
#define BIT_IS_SET(value, pos) (BIT_ONE(pos) & (value) != 0)
#endif

/**
 * @brief Checks that bit is clear.
 *
 * @param value Value that for test bit.
 * @param pos   Position number of the bit.
 *
 * @return True if this bit is clear, else in otherwise.
 */
#ifndef BIT_IS_CLR
#define BIT_IS_CLR(value, pos) (BIT_ONE(pos) & (value) == 0)
#endif

/**
 * @brief Indicates bits changed between 0 <-> 1.
 *
 * @param before    Bits state before changing.
 * @param before    Bits state after changing.
 *
 * @return Changed bits.
 */
#ifndef BITS_CHANGED
#define BITS_CHANGED(before, after) ((before) ^ (after))
#endif

/**
 * @brief Indicates bits changed 0 -> 1.
 *
 * @param before    Bits state before changing.
 * @param before    Bits state after changing.
 *
 * @return Bits are rise up.
 */
#ifndef BITS_RISING
#define BITS_RISING(before, after) (BITS_CHANGED(before, after) & (after))
#endif

/**
 * @brief Indicates bits changed 1 -> 0.
 *
 * @param before    Bits state before changing.
 * @param before    Bits state after changing.
 *
 * @return Bits are fall down.
 */
#ifndef BITS_FALLING
#define BITS_FALLING(before, after) (BITS_CHANGED(before, after) & (before))
#endif

/**
 * @brief Macro for performing rounded integer division (as opposed to truncating the result).
 *
 * @param   A   Numerator.
 * @param   B   Denominator.
 *
 * @return      Rounded (integer) result of dividing A by B.
 */
#ifndef ROUNDED_DIV
#define ROUNDED_DIV(A, B) (((A) + ((B) / 2)) / (B))
#endif

/**
 * @brief Macro for counting items in an object.
 *
 * @param   x   An object for which the counting will be made.
 *
 * @return      A number of items in an object.
 */
#ifndef COUNT_OF
#define COUNT_OF(x) ((sizeof(x) / sizeof(0 [x])) / ((size_t)(!(sizeof(x) % sizeof(0 [x])))))
#endif

/**
 * @brief Macro for getting offset for a field in the provided type.
 *
 * @param   type   Provided type.
 * @param   field  Field in the provided type.
 *
 * @return         Offset of a field in the provided type.
 */
#ifndef OFFSET_OF
#define OFFSET_OF(type, field) ((unsigned long)&(((type*)0)->field))
#endif

#endif /* _NRF_EDGEAI_MACRO_H_ */