/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */

/**
 * The APIs in this header file are for the pre-compiled driver library. Users should avoid these low-level APIs
 * in favor of higher level APIs in nrf_axon_nn_infer.h and nrf_axon_infer_test.h
 */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>


#if !defined(AXON_FORCE_32BIT_ADDR) && ((defined(__SIZEOF_POINTER__) && (__SIZEOF_POINTER__==8)) || defined(_WIN64))
typedef uint64_t NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE;
typedef int64_t NRF_AXON_PLATFORM_BITWIDTH_SIGNED_TYPE;
#else
#error
typedef uint32_t NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE;
typedef int32_t NRF_AXON_PLATFORM_BITWIDTH_SIGNED_TYPE;
#endif

/**
 * @brief Axon driver return codes
 *
 * Axon driver functions return either a positive success value
 * or a negative error code.
 */
typedef enum {
  NRF_AXON_RESULT_MUTEX_FAILED                  = -203, /**< Unable to acquire the mutex to access the axonpro */
  NRF_AXON_RESULT_INVALID_MODEL                 = -202, /**< problem with the model structure. */
  NRF_AXON_RESULT_BUFFER_TOO_SMALL              = -18,  /**< Buffer(s) are too small to perform the operation. */
  NRF_AXON_RESULT_INVALID_CMD_BUF               = -17,  /**< Cmd buffer is invalid */
  NRF_AXON_RESULT_FAILURE_MISSING_NULL_COEF     = -14,  /**< FIR requires last filter coefficient to be 0. */
  NRF_AXON_RESULT_NULL_BUFFER                   = -13,  /**< one or more required buffers is NULL */
  NRF_AXON_RESULT_FAILURE_INVALID_ROUNDING      = -12, /**< Rounding value specified is out of range */
  NRF_AXON_RESULT_FAILURE_MISALIGNED_BUFFER     = -7,  /**< One or more buffers do not meet alignment requirements for the requested data_width  */
  NRF_AXON_RESULT_FAILURE_UNSUPPORTED_HARDWARE  = -6, /**< axon driverhardware version is unsupported */
  NRF_AXON_RESULT_FAILURE_HARDWARE_ERROR        = -5, /**< error within axon hardware */
  NRF_AXON_RESULT_FAILURE_INVALID_LENGTH        = -2, /**< length provided in the was invalid */
  NRF_AXON_RESULT_FAILURE                       = -1, /**< generic failure code */
  NRF_AXON_RESULT_SUCCESS                       = 0, /**< success */
  NRF_AXON_RESULT_NOT_FINISHED                  = 1,
  NRF_AXON_RESULT_EVENT_PENDING                 = 3, /**< return by handle_interrupt. The interrupt resulted in an event that needs to be processed. */
} nrf_axon_result_e;

/**
 * @brief
 * Internal structure supplied by user but managed by the driver to track execution progress.
 */
typedef struct {
  uint32_t length;                                              /**< # of elements in cmd_buf_ptr[] */
  const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* cmd_buf_ptr;  /**< command buffer of axon compiled code */
  uint32_t cur_segment_offset;                                  /**< managed by the driver to track progress */
} nrf_axon_cmd_buffer_info_s;

/**
 * @brief Specifies the blocking mechanism for synchronous Axon command buffer execution
 *
 * nrf_axon_run_cmd_buf_sync() provides a synchronous interface to executing an Axon command buffer.
 * The optimal blocking scheme is dependent on the work load being presented.
 *
 * Smaller work loads like intrinsics are faster and more energy efficient when a hardware status polling
 * loop is used, because the overhead of interrupt handling is high relative to the Axon work load.
 *
 * Larger work loads like NN model inference will be more energy efficient using interrupts because the CPU
 * can sleep during Axon execution. The interrupt processing overhead is small compared to the Axon
 * execution time.
 *
 * A potential future option is to defer blocking and return immediately and allow the caller to proceed with other work,
 * then perform the wait with a call to a TBD function. Callers must ensure that no variables passed to
 * nrf_axon_run_cmd_buf_sync fall out of scope, and that there are no interdependencies between the work Axon is doing
 * and the work the CPU is doing.
 */
typedef enum {
  NRF_AXON_SYNC_MODE_BLOCKING_INVALID,    /**< DO NOT USE! Reserved for driver use. */
  NRF_AXON_SYNC_MODE_BLOCKING_POLLING,    /**< Block by waiting in a software polling loop. Appropriate for intrinsics where the overhead of interrupt processing is high compared to the Axon compute load. */
  NRF_AXON_SYNC_MODE_BLOCKING_EVENT,      /**< Block by waing on a signal. Axon hardware completion will fire an interrupt, and the ISR will signal completion. Appropriate for large work loads like nn model inference. */
  NRF_AXON_SYNC_MODE_BLOCKING_DEFERRED,   /**< DO NOT USE! UNTESTED! don't wait for hardware to complete. User will call wait function later. */
  NRF_AXON_SYNC_MODE_BLOCKING_COUNT,      /**< Reserved for driver use. Allows arrays to be sized by the length of the enum. */
} nrf_axon_syncmode_blocking_e;


/**
 * @brief Used as a parameter to nrf_axon_queue_cmd_buf for asynchronous execution.
 */
typedef struct nrf_axon_queued_cmd_info_wrapper_s {
  nrf_axon_cmd_buffer_info_s *cmd_buf_info;             /**< pointer to the cmd buffer to execute */
  void* callback_context;                               /**< caller-provided pointer that is provided in the callback function. */
  void (*callback_function)(nrf_axon_result_e result, void* callback_context);  /**< caller-provided function to be invoked when the operation list is completed */
  const int8_t *input_vector;                           /**< If not NULL, input data will be copied from here to input_buffer immediately prior to execution. Needed if there is any possibility axon is in use by any other user.  */
  int8_t *input_buffer;                                 /**< Location of input as compiled into the command buffer. */
  uint16_t input_size;                                  /**< size in bytes of the input to be copied from input_vector to input_buffer. */
  int8_t *tmp_output_buffer;                            /**< if not null, driver will copy the results from here. */
  int8_t *output_buffer;                                /**< if not null, driver will copy the results here*/
  uint16_t output_width_in_bytes;                       /**< width of an output row in units of bytes. */
  uint16_t output_stride;                               /**< distance between rows of output in units of bytes, >= output_width_in_bytes*/
  uint16_t output_buffer_packed_size;                   /**< total size of the packed output in bytes. */
  struct nrf_axon_queued_cmd_info_wrapper_s* next;      /**< Managed by the driver to place this entry in a linked-list queue. */
} nrf_axon_queued_cmd_info_wrapper_s;

/**
 * @brief Asynchronous command buffer execution
 *
 * Low level function that adds a command buffer to the Axon command buffer queue to be executed at the next opportunity.
 * Operates asynchronously; returns as soon as the command buffer has been added to the queue. Caller supplied call back
 * function in cmd_buf_wrapper will be invoked upon completion.
 *
 * For asynchronous AI Model inference, users should call the higher level function nrf_axon_nn_model_infer_async.
 *
 * Caller is responsible for populating cmd_buf_wrapper, but driver manages it.
 * @param[in] cmd_buf_wrapper wrapper struct that includes the command buffer and a callback function.
 * Must allocated from static memory and remain valid until the callback function is invoked. (ie, can't be
 * placed on the stack). Caller populates most of this structure, but driver manages
 */
nrf_axon_result_e nrf_axon_queue_cmd_buf(nrf_axon_queued_cmd_info_wrapper_s* cmd_buf_wrapper);

/**
 * @brief Synchronous command buffer execution
 *
 * Low level function that executes the command buffer in cmd_buf_info on Axon in a way that
 * appears synchronous to the user. It returns when the execution is complete.
 * Waits for exclusive access to axon using nrf_axon_platform_reserve_for_user(),
 * then executes the command buffer in cmd_buf_info synchronously.
 * The caller can specify to "keep" the reservation in case a series of axon command buffers are
 * executed in succession.
 * The reservation can be returned on the last command buffer or explictly by the user with
 * nrf_axon_platform_free_reservation_from_user().
 *
 * @param[in] cmd_buf_info command buffer that has been initialized by nrf_axon_init_command_buffer_info.
 * @param[in] block_mode Specifies how to wait for Axon completion.
 * @param[in] keep_reservation If true, Axon reservation is not freed before returning; user maintains ownership of Axon.
 *
 * @retval 0 on success or a negative error code.
 */
nrf_axon_result_e nrf_axon_run_cmd_buf_sync(
  nrf_axon_cmd_buffer_info_s* cmd_buf_info,
  nrf_axon_syncmode_blocking_e block_mode,
  bool keep_reservation);

/**
 * @brief binds a command buffer to an nrf_axon_cmd_buffer_info_s struct and initializes it.
 * Need only be called once per command buffer.
 * @param[out] cmd_buf_info_ptr pointer to the struct that will be bound to.
 * @param cmd_buf buffer containing compiled Axon code.
 * @param buffer_length length of cmd_buf in units of elements, not bytes.
 * @return kAxonResultSuccess if success or a negative error code.
 */
nrf_axon_result_e nrf_axon_init_command_buffer_info(
  nrf_axon_cmd_buffer_info_s* cmd_buf_info_ptr,
  const NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* cmd_buf,
  uint32_t buffer_length);

#ifdef __cplusplus
}
#endif
