/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */

 /**
 * Functions used by Axon platform abstraction (nrf_axon_platform) code to interact with the Axon driver.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include "nrf_axon_driver.h"

/**
 * @brief one-time driver initialziation code.
 * 
 * Platform code must invoke this function prior to any other use of Axon.
 * Axon must be powered-on and have clocks enabled.
 * 
 * @param[in] base_address Base address of the Axon IP block. Obtained through device tree on Zephyr systems.
 * @retval 0 on success are a negative error code.
 */
nrf_axon_result_e nrf_axon_driver_init(void *base_address);

/**
 * @brief Enables Axon after being powered on
 * 
 * @note Must not be called before nrf_axon_driver_init().
 * @retval 0 on success are a negative error code.
 */
nrf_axon_result_e nrf_axon_driver_power_on();

/**
 * @brief Clean-up code invoked before Axon is powered off.
 * 
 * @note Must not be called before nrf_axon_driver_init().
 * @retval 0 on success are a negative error code.
 */
nrf_axon_result_e nrf_axon_driver_power_off();

/**
 * @brief Low level Axon interrupt handler
 *
 * This function is called by the platform interrupt handler to check for and handle an Axon interrupt.
 * This function will clear the interrupt at its source so that system interrupts can be safely re-enabled.
 * 
 * If further processing is required, it invokes nrf_axon_platform_generate_driver_event(), which
 * should lead to the platform calling nrf_axon_process_driver_event() either indirectly (by signalling the Axon driver thread on an RTOS system)
 * or directly (in bare-metal systems).
 * 
 * @retval kAxonResultEventPending Indicates that nrf_axon_process_driver_event() was called.
 */
nrf_axon_result_e nrf_axon_handle_interrupt();

/**
 * @brief Called by platform in response to nrf_axon_generate_driver_event()
 * 
 * @retval 0 on success are a negative error code.
*/
nrf_axon_result_e nrf_axon_process_driver_event();

/**
 * @brief Indicates if any asynchronous inference jobs are in the queue.
 * 
 * returns true if the queue is empty and nrf_axon_start_queue_processing() needs to be called.
 * This needs to happen when a user frees axon reservation to kick-start asynchronous queue processing, because
 * the driver does not wait if axon_platform_reserve_for_driver() fails.
 * 
 * @retval[false] No async jobs pending
 * @retval[true] 1 or more async jobs pending
 */
bool nrf_axon_queue_not_empty();

/**
 * @brief Initiates asynchronous operation after synchronous operation completes
 * 
 * Any asynchronous jobs submitted to the Axon driver are stalled if a synchronous job is in process.
 * Upon completion of the synchronous job, driver will invoke nrf_axon_platform_free_reservation_from_user().
 * That function must call nrf_axon_queue_not_empty() to see if there are pending async jobs. If so, it gives
 * owenership directly to the driver (without releasing the ownership semaphore) then calls nrf_axon_start_queue_processing().
 */
void nrf_axon_start_queue_processing();


#ifdef __cplusplus
}
#endif
