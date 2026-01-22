/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "nrf_axon_driver.h"

/**
 * @brief Axon dsp intrinsic vector lengths must be a multiple of 4.
 */
#define NRF_AXON_DSP_VECTOR_MULTPLE (4)
#define NRF_AXON_DSP_VECTOR_LENGTH_CEIL(vector_length) (NRF_AXON_DSP_VECTOR_MULTPLE * (((vector_length)+NRF_AXON_DSP_VECTOR_MULTPLE-1)/NRF_AXON_DSP_VECTOR_MULTPLE))

/**
 * Output Rounding
 * Rounding will shift the output values by the number of bits
 * specified, then add 1 if the shifted MSB is a 1.
 */
typedef enum {
  kAxonRoundingNone = 0,  /**< numbers are not rounded   */
  kAxonRoundingMax = 32 /**< Maximum supported rounding value */
} AxonRoundingEnum;

/**
 * @brief Performs a 24bit (unpacked) complex FFT. 
 * complex numbers are stored in pairs, real component, imaginary component. 
 * length_log2 is log2(fft length). (ie, fft length=1<<length_log2).
 * For a real FFT, imaginary coefficients must be set to 0.
 * @param in_ptr 32bit aligned pointer to the complex input. length=2*(1<<length_log2) 
 * @param out_ptr 32bit aligned pointer to the output.. length=2*(1<<length_log2). output can overlap input.
 * @param length_log2 number of complex numbers log2.
 * @param half_output true if only the 1st half of the fft output should be written (ie, below nyquist frequency)
 */
nrf_axon_result_e axon_fft_24(
  const int32_t* in_ptr,
  int32_t *out_ptr,
  uint16_t length_log2,
  bool half_output,
  bool round_by_length_log2,
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief Performs a 24bit (unpacked) complex FFT, then sums the squares of the complex coefficients,
 * and divides by fft length.
 * complex numbers are stored in pairs, real component, imaginary component. 
 * length_log2 is log2(fft length). (ie, fft length=1<<length_log2).
 * For a real FFT, imaginary coefficients must be set to 0.
 * @param in_ptr 32bit aligned pointer to the complex input. length=2*(1<<length_log2) 
 * @param out_ptr 32bit aligned pointer to the output.. length=(1<<length_log2). output can overlap input.
 * @param length_log2 number of complex numbers log2.
 * @param half_output true if only the 1st half of the fft output should be written (ie, below nyquist frequency)
 * @param rounding_bits Any additional rounding that should be performed on the output.
 */
nrf_axon_result_e axon_fft_power_24(
  const int32_t* in_ptr, 
  int32_t *out_ptr, 
  uint16_t length_log2, 
  bool half_output, 
  uint8_t rounding_bits,
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief "FIR" => "Finite Impulse Response" filter. Filter coefficients are MAC'd with the input in reverse order.
 * ie, for filter length F, output x[n] input[n-F]*filter[F-1] + input[n-F-1]*filter[F-2]+...input[n-1]*filter[0].
 * The filter output starts at outptr+ filter_length. out_ptr[0..filter_length-1] are written to but should be discarded.
 * 
 * @param in_put pointer to the input, 24bit integer sign-extended to 32bits.
 * @param input_length length of the input (in elements). Must be a multiple of 4, maximum of 1024, and be at least 4 greater than filter_length
 * @param filter_ptr pointer to the filter coefficients, 24bit integer sign-extended to 32bits. 
 * @param filter_length length of the filter (in elements). Must be a at least 12, a multiple of 4 and the last coefficient must be 0. 0 pad as necessary to meet these requirements.
 * @param rounding_bits number of bits to round the output by.
 * @param out_ptr pointer to the output, int24 sign-extended to 32bits. 1st valid output starts at offset filter_length. Output can overlap/overwrite input.
 * 
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_fir_24_24_24(
  const int32_t* in_ptr, 
  const int32_t* filter_ptr, 
  int32_t *out_ptr, 
  uint16_t input_length,
  uint16_t filter_length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief "FIR" => "Finite Impulse Response" filter. Filter coefficients are MAC'd with the input in reverse order.
 * ie, for filter length F, output x[n] input[n-F]*filter[F-1] + input[n-F-1]*filter[F-2]+...input[n-1]*filter[0].
 * The filter output starts at outptr+ filter_length. out_ptr[0..filter_length-1] are written to but should be discarded.
 * 
 * @param in_put pointer to the input, 24bit integer sign-extended to 32bits.
 * @param input_length length of the input (in elements). Must be a multiple of 4, maximum of 1024, and be at least 4 greater than filter_length
 * @param filter_ptr pointer to the filter coefficients, 16bit integer.
 * @param filter_length length of the filter (in elements). Must be a at least 12, a multiple of 4 and the last coefficient must be 0. 0 pad as necessary to meet these requirements.
 * @param rounding_bits number of bits to round the output by.
 * @param out_ptr pointer to the output, int24 sign-extended to 32bits. 1st valid output starts at offset filter_length. Output can overlap/overwrite input.
 * 
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_fir_24_16_24(
  const int32_t* in_ptr, 
  const int16_t* filter_ptr, 
  int32_t *out_ptr, 
  uint16_t input_length,
  uint16_t filter_length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);
/**
 * @brief "Sqrt"=> "Square Root" Calculates a the square root of vector input, output=SQRT(input). Note that the radix of the result is half
 * the radix of the input.
 * 
 * @param in_ptr pointer to the input, 24bit integer sign-extended to 32bits.
 * @param length length of the input (in elements). Must be a multiple of 2, at least 4, and maximum of 512.
 * @param out_ptr pointer to the output, int24 sign-extended to 32bits. Output can overlap/overwrite input.
 * 
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_sqrt_24(
  const int32_t* in_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief 
 * "logn"=> "Natural Log" Calculates the natural log of vector input, output=ln(input). Input and output are both interpreted as q11.12.
 * If the input radix doesn't match 11.12, then an offset can be added to the output to correct the result for radix 12..
 * For an input radix "r", the offset to add is ln(2^^r) << 12.
 * 
 * @param input pointer to the input, q11.12 sign-extended to 32bits.
 * @param input_length length of the input (in elements). Must be a multiple of 2, at least 2, and maximum of 512.
 * @param output pointer to the output, q11.12  sign-extended to 32bits. Output can overlap/overwrite input.
 * @param cmd_buf_info_ptr compiled axon code is appended into this structure. 
 * 
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_logn_11p12(
  const int32_t* in_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief 
 * "exp"=> "exponent" (e to the x) Calculates the exponent of vector input, output=exp(input). Input and output are both interpreted as q11.12.
 * If the input radix isn't 11.12, there is no analogous way to logn() to correct the output. (ie, one can't add or multiple by an offset to correct
 * the result).
 * The effective input range (without saturation) is -8.3 to 7.46
 *
 * @param input pointer to the input, q11.12 sign-extended to 32bits.
 * @param input_length length of the input (in elements). Must be a multiple of 2, at least 2, and maximum of 512.
 * @param output pointer to the output, q11.12  sign-extended to 32bits. Output can overlap/overwrite input.
 * @param cmd_buf_info_ptr compiled axon code is appended into this structure. 
 * 
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_exp_11p12(
  const int32_t* in_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Xspys"=>"X Squared Plus Y Squared" adds the square of vector Y to the square of vector X.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_xspys_24_24_24(
  const int32_t *x_ptr, 
  const int32_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Xspys"=>"X Squared Plus Y Squared" adds the square of vector Y to the square of vector X.
 * input_stride2 indicates that the input vectors will skip every other entry. This is useful for
 * calculating fft power, as the x vector can be the real components, and the y vector can by the imaginary components.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_xspys_24_24_24_input_stride2(
  const int32_t *x_ptr, 
  const int32_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Xsmys"=>"X Squared Minus Y Squared" subtracts the square of vector Y from the square of vector X.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_xsmys_24_24_24(
  const int32_t *x_ptr, 
  const int32_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Xty"=>"X Times Y "  multiplies vector X and vector Y.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_xty_24_24_24(
  const int32_t *x_ptr, 
  const int32_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Xpy"=>"X  Plus Y " vector adds vector X and vector Y.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_xpy_24_24_24(
  const int32_t *x_ptr, 
  const int32_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);


/**
 * @brief
 * "Xmy"=>"X Minus Y" subtracts vector Y from vector X.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_xmy_24_24_24(
  const int32_t *x_ptr, 
  const int32_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "axpb"=>"a times x + b" Multiplies eache vector value by "a" and adds "b"
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param a_scalar Pointer to "a" scalar value. 24bit integers unpacked.
 * @param b_scalar Pointer to "b" scalar value. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_axpb_24_24( 
  const int32_t *x_ptr, 
  const int32_t *a_scalar, 
  const int32_t *b_scalar, 
  int32_t *output, 
  uint16_t length, 
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "axpby"=>"a times x + b times y" Multiplies each x vector value by "a" and adds the product of "b" and the corresponding 
 * y vector value.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param a_scalar Pointer to "a" scalar value. 24bit integers unpacked.
 * @param b_scalar Pointer to "b" scalar value. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap either or both inputs.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_axpby_24_24_24(
  const int32_t* x_ptr,
  const int32_t* y_ptr,
  const int32_t* a_scalar, 
  const int32_t* b_scalar, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "xs"=>"x squared" Each vector value is multiplied by itself
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param out_ptr output vector. 24bit integers unpacked, no extra stride, and rounded by rounding_bits. output can overlap input.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 2, at least 4, and no greater than 512. Does not include the stride.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_xs_24_24(
  const int32_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Mar" => "Multiply, Accumulate, recursive" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_mar_24_24_32(
  const int32_t* x_ptr, 
  const int32_t* y_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Mar" => "Multiply, Accumulate, recursive" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_mar_16_24_32(
  const int16_t* x_ptr, 
  const int32_t* y_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Mar" => "Multiply, Accumulate, recursive" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param y_ptr y vector input. 16bit integers packed.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_mar_16_16_32(
  const int16_t* x_ptr, 
  const int16_t* y_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Mar" => "Multiply, Accumulate, recursive" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 24bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_mar_24_24_24(
  const int32_t* x_ptr, 
  const int32_t* y_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Mar" => "Multiply, Accumulate, recursive" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param y_ptr y vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 24bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_mar_16_24_24(
  const int16_t* x_ptr, 
  const int32_t* y_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Mar" => "Multiply, Accumulate, recursive" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param y_ptr y vector input. 16bit integers packed.
 * @param out_ptr output scalar. 24bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_mar_16_16_24(
  const int16_t* x_ptr, 
  const int16_t* y_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Marx" => "Multiply, Accumulate, recursive, only reload the x vector" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * The y vector must have been pre-loaded using a Mar operation immediately before the 1st Marx.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_marx_24_32(
  const int32_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Marx" => "Multiply, Accumulate, recursive, only reload the x vector" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * The y vector must have been pre-loaded using a Mar operation immediately before the 1st Marx.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_marx_16_32(
  const int16_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Marx" => "Multiply, Accumulate, recursive, only reload the x vector" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * The y vector must have been pre-loaded using a Mar operation immediately before the 1st Marx.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 24bit integer unpacked, rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_marx_24_24(
  const int32_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Marx" => "Multiply, Accumulate, recursive, only reload the x vector" sums the products of elements in input vector X and input vector Y (ie, dot product)
 * The y vector must have been pre-loaded using a Mar operation immediately before the 1st Marx.
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param out_ptr output scalar. 32bit integer saturated at 24bits, rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_marx_16_24(
  const int16_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);


/**
 * @brief
 * "Acc" => "Accumulate" sums the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_acc_24_32(
  const int32_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Acc" => "Accumulate" sums the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_acc_16_32(
  const int16_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Acc" => "Accumulate" sums the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 24bit integer unpacked, rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_acc_24_24(
  const int32_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "Acc" => "Accumulate" sums the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param out_ptr output scalar. 32bit integer saturated at 24bits, rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_acc_16_24(
  const int16_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "l2norm" => sums the square of the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_l2norm_24_32(
  const int32_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "l2norm" => sums the square of the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param out_ptr output scalar. 32bit integer rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_l2norm_16_32(
  const int16_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "l2norm" => sums the square of the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 24bit integers unpacked.
 * @param out_ptr output scalar. 24bit integer unpacked, rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_l2norm_24_24(
  const int32_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * "l2norm" => sums the square of the elements in vector X
 * 
 * @param axon_handle
 *   Handle to a physical axon instance instantiated via AxonInitInstance().
 *
 * @param x_ptr x vector input. 16bit integers packed.
 * @param out_ptr output scalar. 32bit integer saturated at 24bits, rounded by rounding_bits.
 * @param length  Number of elements in the vectors (data samples). Must be a multiple of 4, at least 8, and no greater than 1024.
 * @param rounding_bits  Number of bits to round the output by. 0 => no rounding, maximum of 31.
 *
 * @return kAxonResultSuccess on success or a negative error code (see nrf_axon_result_e)
 */
nrf_axon_result_e axon_l2norm_16_24(
  const int16_t* x_ptr, 
  int32_t *out_ptr, 
  uint16_t length,
  uint8_t rounding_bits, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * vector multiplies 2 16bit vectors to produce a 32bit vector
 * 
 * @parameter out_ptr location to place output. Can safely overlap either input
 * @parameter length number of of elements in the input vectors. Maximum value is 512.
 * @rounding_bits number of bits to round the output by.
 */
nrf_axon_result_e axon_xty_16_16_32(
  const int16_t* x_ptr, 
  const int16_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * vector multiplies 2 16bit vectors to produce a 32bit vector, with option to have unpacked output vector.
 * 
 * Useful for applying a hamming window on real input before performing an 
 * FFT, which requires complex input. Pass output_extra_stride=1 and the output will be placed every
 * other word in the real coefficients slots. A separate function would need to be called
 * to 0 out the imaginary coefficients.
 * 
 * If output_extra_stride is not needed it can be set to 0, but using axon_xty_16_16_32()
 * is preferred for better performance.
 * It performs multiplication of 2 16bit vectors and produces a 32bit output.
 * @parameter out_ptr location to place output. Can safely overlap either input
 * @parameter length number of of elements in the input vectors. Maximum value is 512.
 * @rounding_bits number of bits to round the output by.
 * @parameter output_extra_stride Nuumber of output slots to skip between each output value (skipped slots are not modified)
 */
nrf_axon_result_e axon_xty_16_16_32_output_stride(
  const int16_t* x_ptr, 
  const int16_t *y_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t rounding_bits,  
  uint8_t output_extra_stride, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * Sets 32bit words with the value in set_val (which is limited to 16bits, but sign extended to 32bits). 
 * 
 * otput_extra_stride allows words to be skipped over. Useful for setting FFT input imaginary coefficients to 0.
 * 
 * If output_extra_stride is not needed it can be set to 0, but using  axon_memset_32()
 * is preferred for better performance.
 * @parameter out_ptr location to place output. Can safely overlap either input
 * @parameter length number of of elements to fill.
 * @parameter output_extra_stride Nuumber of output slots to skip between each output value (skipped slots are not modified)
 */
nrf_axon_result_e axon_memset_32_output_stride(
  int16_t set_val,
  int32_t *out_ptr, 
  uint16_t length, 
  uint8_t output_extra_stride, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * Saturates a 32bit vector to 24bits, unpacked. 
 * Used to convert 32bit output into 24bit input needed by many axon dsp operations.
 */
nrf_axon_result_e axon_saturate_32_24(
  const int32_t *in_ptr, 
  int32_t *out_ptr, 
  uint16_t length, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

/**
 * @brief
 * Saturates a 32bit vector to 8bits, unpacked.  
 * @param in_ptr source of data to saturate
 * @param out_ptr destination of data to saturate. Must be 32bit aligned. Can be the same as in_ptr. Must be sized to a multiple of 4 bytes.
 * @param length number of elements to saturate. If not a multiple of 4, the remaining padding bytes up to a multiple of 4 will be undefined.
 */
nrf_axon_result_e axon_saturate_32_8(
  const int32_t *in_ptr, 
  int8_t *out_ptr, 
  uint16_t length, 
  nrf_axon_syncmode_blocking_e block_mode, 
  bool keep_reservation);

#ifdef __cplusplus
}
#endif
