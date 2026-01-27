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

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API 
#endif

#include "nrf_axon_driver.h"

#if NRF_AXON_INTERLAYER_BUFFER_SIZE
extern uint32_t nrf_axon_interlayer_buffer[NRF_AXON_INTERLAYER_BUFFER_SIZE/sizeof(uint32_t)];
#endif
#if NRF_AXON_PSUM_BUFFER_SIZE
extern uint32_t nrf_axon_psum_buffer[NRF_AXON_PSUM_BUFFER_SIZE/sizeof(uint32_t)];
#endif


/**
 * @brief returns the frequency of the clock used by nrf_axon_platform_get_ticks()
 */
uint32_t nrf_axon_platform_get_clk_hz();

/**
 * @brief Returns the current time in units returned by nrf_axon_platform_get_clk_hz().
 * 
 * Used by test profiling code.
 */
uint32_t nrf_axon_platform_get_ticks();

/**
 * @brief General purpose function for writing to the console.
 */
void nrf_axon_platform_printf(const char *fmt, ...);

/**
 * Peforms necessary one-time-only platform and driver initialization code.
*/
nrf_axon_result_e nrf_axon_platform_init();
void nrf_axon_platform_close();

/**
 * @brief disable/enable all interrupts
 * These functions are used by the driver to perform light-weight synchronization.
 * Interrupts will be disabled to examine and potentially update a state variable that is used
 * across multiple threads and potentially in the interrupt context.
 * 
 * nrf_axon_platform_disable_interrupts() returns the interrupt state immediately prior to interrupts being disabled.
 * The state is then passed as restore_value to nrf_axon_platform_restore_interrupts(restore_value), 
 * which restores the interrupt state.
 */
uint32_t nrf_axon_platform_disable_interrupts();
void nrf_axon_platform_restore_interrupts(uint32_t restore_value);

/** 
 * @brief Reserves Axon hardware use for asynchronous job processing by the driver.
 * 
 * This function must not be called from user code!!!
 * 
 * Non-blocking call to reserve the axons hardware for the driver to service the asynchronous command queue. 
 * The driver will never call this function more than once before calling nrf_axon_platform_reserve_for_driver().
 * The driver will always call nrf_axon_platform_reserve_for_driver() exactly once after calling this function.
 * 
 * @note 
 * This function can only fail if there is a synchronous user that owns the reservation 
 * via a call to nrf_axon_platform_reserve_for_user(). 
 * Since there is no mechanism for the driver to request the reservation again in the future, 
 * the function nrf_axon_platform_free_reservation_from_user()
 * must check with the driver to see if it is waiting for the reservation (via nrf_axon_queue_not_empty()) 
 * and start it if it is (via nrf_axon_start_queue_processing())
 * 
 * @retval true Axon hardware successfully reserved for the driver.
 * @retval false Axon hardware is use by a user, unavailable to the driver.
*/
bool nrf_axon_platform_reserve_for_driver();

/**
 * @brief Frees the Axon hardware reservation made by nrf_axon_platform_reserve_for_driver().
 */
void nrf_axon_platform_free_reservation_from_driver();

/** 
 * @brief Reserves Axon hardware use for synchronous job processing by the user.
 * 
 * This function must not be called from user code!!!
 * 
 * This function is called by the driver on behalf of the user when a synchronous inference is started.
 * Users can call this function in advance if they want to access the interlayer buffer directly, before
 * inference is invoked.
 * Calling this function after inference has completed does not guarantee that the interlayer buffer contents
 * are unchanged.
 * 
 * This is a blocking function. 
 * This function can get called multiple times before nrf_axon_platform_free_reservation_from_user() is called.
 * 
 * @retval true Axon hardware successfully reserved for the user.
 * @retval false Should never happen.
*/
bool nrf_axon_platform_reserve_for_user();


/**
 * @brief Frees the user's Axon reservation made with nrf_axon_platform_free_reservation_from_driver();
 * 
 * Will "kick-start" asynchronous operation if any asynchronous requests occured while Axon was reserved
 * for synchronous use.
 */
void nrf_axon_platform_free_reservation_from_user();

/**
 * @brief Driver event to event processing synchronization function.
 * 
 * The driver invokes this function when it has work to do, in response to an interrupt or
 * an asynchronous processing request.
 * 
 * The platform code must then call nrf_axon_driver_process_event().
 * 
 * In an RTOS system, the platform code signals the driver's thread (also created by the the platform), 
 * which in-turn invokes nrf_axon_driver_process_event().
 * 
 * In a bare-metal system, nrf_axon_driver_process_event() can be called directly from nrf_axon_platform_generate_driver_event().
 */
void nrf_axon_platform_generate_driver_event();

/**
 * @brief Driver to User signaling in synchronous mode.
 * 
 * In synchronous mode, driver will invoke nrf_axon_platform_wait_for_user_event() on behalf of the user to wait for the operation
 * to complete from the user's execution context.
 * 
 * nrf_axon_platform_generate_user_event() will be invoked by the driver (in the drivers execution context)
 * from nrf_axon_process_driver_event() when the user's job is complete. This function must generate the event that 
 * nrf_axon_platform_wait_for_user_event is waiting on.
 * 
 * In an RTOS system, this pair of functions should be implemented with a semaphore with maximum count of 1 and default 0 (available).
 * Bare metal systems would implement this pair of functions with a spinlock.
 */
void nrf_axon_platform_wait_for_user_event();

void nrf_axon_platform_generate_user_event();

/**
 * simulator-only function to iterate through a series of files in a folder.
 * Will allocate buffer for the caller, sized to buffer_size.
 */
int nrf_axon_simulator_run_test_files(
  char* input_file_path, 
  char* output_file_path, 
  char* input_file_ext, 
  char* output_file_head_str, 
  uint32_t buffer_size,
  int (*callback_function)(char* input_file_name, char* output_file_name, int8_t* buffer, uint32_t buffer_size));

#ifdef __cplusplus
}
#endif

