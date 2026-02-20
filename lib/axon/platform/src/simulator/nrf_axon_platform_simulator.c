/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "axon/nrf_axon_platform.h"
#include "drivers/axon/nrf_axon_platform_interface.h"
#include "drivers/axon/nrf_axon_driver.h"
#include "axon/nrf_axon_platform_simulator.h"

uint32_t nrf_axon_interlayer_buffer[NRF_AXON_INTERLAYER_BUFFER_SIZE/sizeof(uint32_t)];
uint32_t nrf_axon_psum_buffer[NRF_AXON_PSUM_BUFFER_SIZE/sizeof(uint32_t)];

AxonCoreSatCntLogSt axon_core_saturation_cnt = {0};
AxonFuncSatLogSt axon_function_saturation_log = {0};


uint8_t* axon_nn_system_memory_ptr = NULL;
bool simulator_in_threadless_mode = false;

void delay_us(uint32_t delay) {
}

// #define MY_LOG_LEVEL LOG_LEVEL_NONE
#define MY_LOG_LEVEL LOG_LEVEL_DBG


void nrf_axon_platform_printf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char tempstring[512]; //increased the buffer length from 255 to 512 to accommodate larger strings
  vsnprintf(tempstring, sizeof(tempstring), fmt, args);
  va_end(args);
  printf("%s", tempstring);
  fflush(NULL);
}

// assume 100Mhz
uint32_t nrf_axon_platform_get_clk_hz() {
  return 100000000;	
}

uint32_t nrf_axon_platform_disable_interrupts() {
  return 0;
}
void nrf_axon_platform_restore_interrupts(uint32_t restore_value) {
}

/**
 * platform function to generate an event that will cause nrf_axon_process_driver_event() to be invoked.
 */
volatile static int user_event_sem = 0;
void nrf_axon_platform_generate_driver_event() {
    nrf_axon_process_driver_event();
}

void nrf_axon_platform_wait_for_user_event() {
    while(user_event_sem==0);
    user_event_sem=0;
}
void nrf_axon_platform_generate_user_event(){
    if (user_event_sem ==0) {
      user_event_sem = 1;
    }
}

/*
 * Registered ISR for axonpro interrupt
 * In our simple application we handle IRQs directly
 */
void host_irq_handler(void * data) {
  /* Axon ISR will invoke nrf_axon_platform_generate_driver_event() if further processing is needed. */
  nrf_axon_handle_interrupt();
}

/**
 * Simulator is "bare metal". There is 1 user thread and the driver runs in the "interrupt"
 * context (a separate thread).
 * To have contention between synchronous and asychronous modes, the user thread would need
 * to start an async operation (submit a queue of command buffers) then immediately reserve the hardware.
 * Either way, it is only the user thread that will ever request to reserve the hardware for
 * either party (driver or user), so we don't have to worry about race conditions when accessing 
 * variables.
 * 
 * In the case of calling intrinsics from cpu ops in the command buffer, the request will be made to
 * reserve hardware for the current "user", which happens to be the driver.
 */
#define AXONS_OWNER_ID_NOONE 0
#define AXONS_OWNER_ID_DRIVER 1
#define AXONS_OWNER_ID_USER 2
static volatile int axons_owner_id = AXONS_OWNER_ID_NOONE; 

/**
 * Increments the power vote. If this is the 1st 1, axon is powered on.
 * Returns the number of outstanding votes.
*/
static int axon_power_vote_cnt = 0;
static int vote_for_power()
{
  if (0==axon_power_vote_cnt++) {
      nrf_axon_driver_power_on();
  }
  return axon_power_vote_cnt;
}
/**
 * decrements the power vote. If this is the last one, axon is powered off.
 * Returns the number of outstanding votes.
*/
static int vote_against_power()
{
  if (0==--axon_power_vote_cnt) {
      nrf_axon_driver_power_off();
  }
  return axon_power_vote_cnt;
}

/**
 * to use the hardware in synchronous mode, must reserve for
 * exclusive use.
 */
bool nrf_axon_platform_reserve_for_user() {
  // simulator platform is "bare-metal", synchronous. No way to tell if the caller is the driver or the user, so just return true
  vote_for_power();
  return true;
}
/**
 * Driver is asynchronous, so no waiting on the semaphore.
 */
bool nrf_axon_platform_reserve_for_driver() {
  // simulator platform is "bare-metal", synchronous. No way to tell if the caller is the driver or the user, so just return true
  vote_for_power();
  return true;
}

/**
 * this only gets called from user threads. If a user is freeing the driver, the driver has priority to
 * get it. Could do this by making the driver a higher priority thread...
 */
void nrf_axon_platform_free_reservation_from_user() {
  vote_against_power();

  if (nrf_axon_queue_not_empty()) {
    // driver needs the axon hardware, so don't free the sem, just start the hardware.
    axons_owner_id = AXONS_OWNER_ID_DRIVER;
    nrf_axon_start_queue_processing();
    return;
  }
  axons_owner_id = AXONS_OWNER_ID_NOONE;
}

void nrf_axon_platform_free_reservation_from_driver() {
  vote_against_power();
  axons_owner_id = AXONS_OWNER_ID_NOONE;
}

volatile bool axon_simulator_ints_enabled = false;
static void disable_axon_interrupt() {
  axon_simulator_ints_enabled = false;
}
static void enable_axon_interrupt() {
  axon_simulator_ints_enabled = true;
}

nrf_axon_result_e nrf_axon_platform_init() {
  void *axon_base_address;
  nrf_axon_platform_printf("simulator_in_threadless_mode = %d\n", simulator_in_threadless_mode);
  axon_base_address = start_simulator();
  nrf_axon_result_e result;
  if (NRF_AXON_RESULT_SUCCESS != (result = nrf_axon_driver_init(axon_base_address))) {
    return result;
  }
  
  enable_axon_interrupt();

  return NRF_AXON_RESULT_SUCCESS;
}

void nrf_axon_platform_close() {
  exit_simulator();
}


int read_in_test_vector_int8(FILE* src_file, int8_t* test_vector_buffer, uint32_t buffer_length) {
  unsigned int result = 0;
  int scalar_val;
  char delimit_char;

  while (result++ < buffer_length) {
    if (1 > fscanf_s(src_file, "%d%c", &scalar_val, &delimit_char, (unsigned int)sizeof(delimit_char))) {
      result--;
      break; // end of file?
    }
    // end of the line
    // fixme! make sure no overflow!
    *test_vector_buffer++ = scalar_val;
    if ((delimit_char == '\r') || (delimit_char == '\n') ){ //handling /r and /n, for files generated in windows/linux systems
      break;
    }
  }
  return result;
}

int read_in_test_vector_int16(FILE* src_file, int16_t* test_vector_buffer, uint32_t buffer_length) {
  unsigned int result = 0;
  int scalar_val;
  char delimit_char;

  while (result++ < buffer_length) {
    if (1 > fscanf_s(src_file, "%d%c", &scalar_val, &delimit_char, (unsigned int)sizeof(delimit_char))) {
      result--;
      break; // end of file?
    }
    // end of the line
    // fixme! make sure no overflow!
    *test_vector_buffer++ = scalar_val;
    if ((delimit_char == '\r') || (delimit_char == '\n') ){ //handling /r and /n, for files generated in windows/linux systems
      break;
    }
  }
  return result;
}

/**
 * Simulator only code.
 * Log the overflow counts together with corrsponding function name
*/
void axon_simulator_log_function_saturation(const char* funcName) {
  AxonCoreSatCntLogSt sat = {0};
  axon_simulator_read_saturation_cnt(&sat);
  size_t funcName_len = strnlen(funcName, 255); //assume max function name length is 255
  for (int i = 0; i < axon_function_saturation_log.functionCount; i++) {
 	  size_t name_in_table_len = strnlen(axon_function_saturation_log.functionTable[i].name, 255);
 	  size_t cmp_len = funcName_len > name_in_table_len ? funcName_len : name_in_table_len;
      if (strncmp(axon_function_saturation_log.functionTable[i].name, funcName, cmp_len) == 0) {
          axon_function_saturation_log.functionTable[i].cnts.overflow_cnt += sat.overflow_cnt;
          axon_function_saturation_log.functionTable[i].cnts.underflow_cnt += sat.underflow_cnt;
          axon_function_saturation_log.functionTable[i].cnts.total_ops_cnt += sat.total_ops_cnt;
          return;
      }
  }
  if (axon_function_saturation_log.functionCount < MAX_FUNCTIONS_LOG) {
      //combines two operations: memory allocation and string copying. Remember to free later
#if _WIN32 || _WIN64
      // windows gives a warning over strdup      
      axon_function_saturation_log.functionTable[axon_function_saturation_log.functionCount].name = _strdup(funcName);
#else
      // GCC gives a linker error over _strdup
      axon_function_saturation_log.functionTable[axon_function_saturation_log.functionCount].name = strdup(funcName);
#endif
      memcpy(&axon_function_saturation_log.functionTable[axon_function_saturation_log.functionCount].cnts, &sat, sizeof(AxonCoreSatCntLogSt));
      axon_function_saturation_log.functionCount++;
  } else {
      nrf_axon_platform_printf("Error: Function saturation logging table is full % entries!\n", MAX_FUNCTIONS_LOG);
  }
}

/**
 * Simulator only code.
 * Print out saturation statistics per function to console.
*/
void axon_simulator_print_saturation_statistics() {
  if (axon_function_saturation_log.functionCount > 0) {
	  nrf_axon_platform_printf("-------------------------------------------------------------------------------------------------------------\n");
	  nrf_axon_platform_printf("%-40s %20s %20s %20s\n", "axon instrinsics:", "overflowCnt", "underflowCnt", "totalOpsCnt");
	  for (uint16_t i=0; i<axon_function_saturation_log.functionCount; i++) {
      nrf_axon_platform_printf("%-40s %20lu, %20lu, %20lu\n", 
        axon_function_saturation_log.functionTable[i].name, 
        axon_function_saturation_log.functionTable[i].cnts.overflow_cnt, 
        axon_function_saturation_log.functionTable[i].cnts.underflow_cnt, 
        axon_function_saturation_log.functionTable[i].cnts.total_ops_cnt);
      free(axon_function_saturation_log.functionTable[i].name);
	  }
	  nrf_axon_platform_printf("------------------------------------------------------------------------------------------------------------\n");
  }
}

void nrf_axon_platform_set_profiling_gpio()
{}
void nrf_axon_platform_clear_profiling_gpio()
{}

volatile static bool axon_perfmodel_enabled = false;
uint64_t nn_o_cycles = 0;
uint64_t dsp_o_cycles = 0;
void nrf_axon_simulator_perfmodel_init() {
  nn_o_cycles = 0;
  dsp_o_cycles = 0;
}

uint64_t nrf_axon_simulator_perfmodel_get_cycles() 
{
 return nn_o_cycles + dsp_o_cycles;
}

void nrf_axon_simulator_perfmodel_disable() {
  axon_perfmodel_enabled = false;
}
void nrf_axon_simulator_perfmodel_enable() {
  axon_perfmodel_enabled = true;
}

bool nrf_axon_simulator_perfmodel_is_enabled() {
  return axon_perfmodel_enabled;
}
