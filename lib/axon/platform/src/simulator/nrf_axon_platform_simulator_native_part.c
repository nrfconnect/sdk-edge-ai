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
#include "nrf_axon_platform_interface.h"
#include "nrf_axon_platform_simulator.h"

/*
 * Registered ISR for axonpro interrupt
 * In our simple application we handle IRQs directly
 */
void host_irq_handler(void * data) {
  /* Axon ISR will invoke nrf_axon_platform_generate_driver_event() if further processing is needed. */
  nrf_axon_handle_interrupt();
}

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

void nrf_axon_platform_printf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  char tempstring[512]; //increased the buffer length from 255 to 512 to accommodate larger strings
  vsnprintf(tempstring, sizeof(tempstring), fmt, args);
  va_end(args);
  printf("%s", tempstring);
  fflush(NULL);
}

AxonCoreSatCntLogSt axon_core_saturation_cnt = {0};
AxonFuncSatLogSt axon_function_saturation_log = {0};

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
      axon_function_saturation_log.functionTable[axon_function_saturation_log.functionCount].name = strdup(funcName);
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


uint8_t* axon_nn_system_memory_ptr = NULL;
