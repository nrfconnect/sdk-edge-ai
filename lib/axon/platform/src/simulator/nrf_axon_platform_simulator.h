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
#include <stdio.h>
#include "nrf_axon_driver.h"

typedef struct {
  uint64_t overflow_cnt;
  uint64_t underflow_cnt;
  uint64_t total_ops_cnt;
}AxonCoreSatCntLogSt;

typedef struct {
  char *name;
  AxonCoreSatCntLogSt cnts;
}AxonFuncSatLogEntrySt;

#define MAX_FUNCTIONS_LOG 100
typedef struct {
  AxonFuncSatLogEntrySt functionTable[MAX_FUNCTIONS_LOG];
  uint16_t functionCount;
}AxonFuncSatLogSt;

extern volatile bool axon_simulator_ints_enabled;


extern uint64_t nn_o_cycles;
extern uint64_t dsp_o_cycles;

 int axon_simulator_get_failure_code();
void axon_dsp_simulator_write_reg(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* addr, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value);
void axon_nn_simulator_write_reg(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* addr, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value);
NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE axon_dsp_simulator_read_reg(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* addr);
NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE axon_nn_simulator_read_reg(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *addr);
void host_wait_for_interrupt();
void exit_simulator();
uint32_t axon_platform_get_ticks();
void nrf_axon_simulator_perfmodel_init();
uint64_t nrf_axon_simulator_perfmodel_get_cycles();
void nrf_axon_simulator_perfmodel_disable();
void nrf_axon_simulator_perfmodel_enable();
bool nrf_axon_simulator_perfmodel_is_enabled();

/*
 starts the simulator and returns the "base address" of axon .
 */
void *start_simulator();

#if __unix || (__APPLE__ && __MACH__)
int fopen_s(FILE** f, const char* name, const char* mode);
int fprintf_s(FILE * stream,const char * format, ...);
int fscanf_s(FILE * stream,const char * format, ...);
size_t fread_s(void * buffer_ptr, size_t buffer_size, size_t size, size_t n, FILE * stream);
char *strcpy_s(char* dest, size_t size, char const* src);
#endif

/*
* register write to axon nn primative.
* returns:
* 0 if normal register written.
* 1 if an "action" register was written (meanning the command buffer needs to be serviced)
* or a negative error code (invalid address or read-only register)
*/
extern int axon_nn_simulator_write_reg_prim(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *addr, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value);

extern int axon_dsp_simulator_write_reg_prim(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* addr, NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE value);

extern NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE axon_dsp_simulator_read_reg_prim(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE* addr);

extern NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE axon_nn_simulator_read_reg_prim(volatile NRF_AXON_PLATFORM_BITWIDTH_UNSIGNED_TYPE *addr);

extern bool axon_nn_simualtor_int_pending_prim();

extern bool axon_dsp_simualtor_int_pending_prim();

/*
* processes a pending axon_dsp action request.
* returns 1 if any interrupts are pending upon completion, 0 if not.
*/
extern int axon_dsp_simulator_process_action_request(int *check_cfg_err, uint64_t *o_cycles, int *o_wdog_cmd, int *o_wdog_finish);

extern int axon_nn_simulator_process_action_request(int *check_cfg_err, uint64_t *o_cycles, int *o_wdog_cmd, int *o_wdog_finish);

/*
  * initializes register values to hardware defaults.
  * returns point to base address of register space.
  */
extern void *axon_dsp_initialize_registers();
extern void *axon_nn_initialize_registers();

// create a register buffer for axons to target when writing to each other
typedef uint32_t AXONS_SIMULATOR_FAKE_REGISTERS[0x400];
extern AXONS_SIMULATOR_FAKE_REGISTERS axons_simulator_fake_registers;

int read_in_test_vector_int8(FILE* src_file, int8_t* test_vector_buffer, uint32_t buffer_length);
int read_in_test_vector_int16(FILE* src_file, int16_t* test_vector_buffer, uint32_t buffer_length);
extern void axon_platform_printf(const char* fmt, ...);

void axon_simulator_log_function_saturation(const char* funcName);
void axon_simulator_print_saturation_statistics();
void axon_simulator_read_saturation_cnt(AxonCoreSatCntLogSt *);
void axon_simulator_clear_saturation_cnt();
nrf_axon_result_e axon_platform_init(void);
void axon_platform_close(void);

#ifdef __cplusplus
}
#endif
