/*
 * Copyright (c) 2024, Nordic Semiconductor ASA. All Rights Reserved.
 *
 * The information contained herein is confidential property of Nordic Semiconductor ASA.
 * The use, copying, transfer or disclosure of such information is prohibited except by
 * express written agreement with Nordic Semiconductor ASA.
 */
#include "nrf_axon_platform_simulator.h"
#include "nrf_axon_platform.h"
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <time.h>
#include <semaphore.h>
#define _GNU_SOURCE // for pthread_timedjoin_np
extern int pthread_timedjoin_np(pthread_t thread, void **retval,
                         const struct timespec *abstime);
#include <pthread.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <dirent.h>
#include <stdlib.h>
#include <sys/stat.h>

volatile bool axon_simulator_ints_enabled;

extern void host_irq_handler(void * data);
/**
 * Structure of global variables pertaining to simulator
 */
typedef struct {
  pthread_mutex_t registers_mutex;                        // mutex for accessing axon pro registers
  int mutex_wait_result;                                // variable to hold mutex wait result
  sem_t semaphore;                              // binary semaphore to simulate axon pro interrupts
  pthread_t hw_thread_handle;                       // thread for axon_pro hardware
  pthread_t int_thread_handle;                       // thread for axon_pro int
  sem_t terminate_int_thread_semaphore;
  volatile bool terminate_thread;
  volatile uint8_t action_reg_updated;                    // indicates that user has modified a command or sticky bit register
} sSIMULATOR_STATE_AXONS;

typedef struct {
  sSIMULATOR_STATE_AXONS axon_dsp;
  sSIMULATOR_STATE_AXONS axon_nn;
} sSIMULATOR_STATE;
static sSIMULATOR_STATE simulator_state;

/*
 * Axon write to register (executes in the application thread only)
 */
void axon_dsp_simulator_write_reg(volatile uint64_t* addr, uint64_t value){

  // grab mutex by calling wait()
  simulator_state.axon_dsp.mutex_wait_result = pthread_mutex_lock(&simulator_state.axon_dsp.registers_mutex);

  // catch failed wait()
  if (simulator_state.axon_dsp.mutex_wait_result != 0) {
    printf("AXON DSP Mutex wait failed..");
    system("PAUSE");
  }

  switch(axon_dsp_simulator_write_reg_prim(addr, value)) {
    case 0: break;
    case 1: simulator_state.axon_dsp.action_reg_updated = true; break;
    case 2: sem_post(&simulator_state.axon_dsp.semaphore); break;
  }

  pthread_mutex_unlock(&simulator_state.axon_dsp.registers_mutex);
}

/**
 * Axon read from register (executes in the application thread only)
 */
uint64_t axon_dsp_simulator_read_reg(volatile uint64_t* addr) {

  // grab mutex by calling wait()
  simulator_state.axon_dsp.mutex_wait_result = pthread_mutex_lock(&simulator_state.axon_dsp.registers_mutex);

  // catch failed wait()
  if (simulator_state.axon_dsp.mutex_wait_result != 0) {
    printf("AXON DSP Mutex wait failed..");
    system("PAUSE");
  }

  // read value from application register set
  uint64_t return_value = axon_dsp_simulator_read_reg_prim(addr);
  pthread_mutex_unlock(&simulator_state.axon_dsp.registers_mutex);
  return return_value;
}
bool axon_dsp_simualtor_int_pending() {
  if (0 != pthread_mutex_lock(&simulator_state.axon_dsp.registers_mutex)) {
    printf("AXON DSP Mutex wait failed..");
    system("PAUSE");
  }
  bool result = axon_dsp_simualtor_int_pending_prim();
  pthread_mutex_unlock(&simulator_state.axon_dsp.registers_mutex);
  return result;
}

void* axon_dsp_int_thread(void* data) {
  bool int_fired = false;
  while (1) {
    if(int_fired || (sem_trywait(&simulator_state.axon_dsp.semaphore)==0)){
      int_fired = true;
      // allow for the int firing while disabled, then the int is cleared. Need to suppress the ISR in that case.
      if (axon_simulator_ints_enabled) {
        if (axon_dsp_simualtor_int_pending()) {
            // call the handler
            host_irq_handler(NULL);
        }
        // only clear  int_fired if the int bits were cleared.
        int_fired = axon_dsp_simualtor_int_pending();
      }
    }
    if(sem_trywait(&simulator_state.axon_dsp.terminate_int_thread_semaphore)==0){
      break;
    }
  }
  return 0;
}


// Axon hardware thread
void* axon_dsp_hw_thread(void* data) {
  // register polling loop

  while (1) {

    if (simulator_state.axon_dsp.terminate_thread) {
      break;
    }

    // wait for driver to write something of interest...
    if (simulator_state.axon_dsp.action_reg_updated == 0) {
      continue; // ...nothing yet, skip below and go back to the beginning of while loop
    }

    // action is necessary - grab mutex by calling wait()
    simulator_state.axon_dsp.mutex_wait_result = pthread_mutex_lock(&simulator_state.axon_dsp.registers_mutex);

    // catch failed wait()
    if (simulator_state.axon_dsp.mutex_wait_result != 0) {
      printf("AXON DSP Mutex wait failed..");
      system("PAUSE");
    }

    // clear out action indicator
    simulator_state.axon_dsp.action_reg_updated = 0;
    // Release axon_dsp registers mutex
    pthread_mutex_unlock(&simulator_state.axon_dsp.registers_mutex);

    /* check interrupts*/
    int dsp_simulator_err, dsp_o_wdog_cmd, dsp_o_wdog_finish;
    if (axon_dsp_simulator_process_action_request(&dsp_simulator_err, &dsp_o_cycles, &dsp_o_wdog_cmd, &dsp_o_wdog_finish)) {
      sem_post(&simulator_state.axon_dsp.semaphore);
    }
  }
  return 0;
}

/**
 * ISR FUNCTION
 */

static int start_simulator_axon_dsp() {

  simulator_state.axon_dsp.terminate_thread = false;

  int ret=0;
  // create mutex
  ret = pthread_mutex_init(&simulator_state.axon_dsp.registers_mutex,NULL);
  if (ret!=0) {
    printf("AXON DSP CreateMutex error: %d\n", ret);
    return -1;
  }
  // create semaphore
  ret = sem_init(&simulator_state.axon_dsp.semaphore,0,0);
  if (ret!=0) {
    printf("AXON DSP CreateSemaphore error: %d\n",ret);
    system("PAUSE");
  }

  // create interrupt thread semaphore
  ret = sem_init(&simulator_state.axon_dsp.terminate_int_thread_semaphore,0,0);
  if (ret!=0) {
    printf("AXON DSP CreateSemaphore interrupt thread terminate error: %d\n", ret);
    system("PAUSE");
  }

  // initialize register values
  axon_dsp_initialize_registers();

  // create threads for axon dsp hardware and the axon dsp interrupt
  /**
   * @FIXME! Check axon cmodel stack size requirements ~64kB
   * EC  - find where stack size is being set/overwritten in exe
   */
  ret = pthread_create(&simulator_state.axon_dsp.hw_thread_handle,NULL,axon_dsp_hw_thread, NULL);
  if (ret!= 0) {
    printf("AXON DSP HW Thread creation failed. Exiting program \n");
    fflush(stdout);
    system("PAUSE");
  }

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec +=1; //wait for 1 seconds
  pthread_timedjoin_np(simulator_state.axon_dsp.hw_thread_handle, NULL, &ts);
  ret = pthread_create(&simulator_state.axon_dsp.int_thread_handle,NULL,axon_dsp_int_thread, NULL);
  if (ret!= 0) {
    printf("AXON DSP INT Thread creation failed. Exiting program \n");
    fflush(stdout);
    system("PAUSE");
  }
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec +=1; //wait for 1 seconds
  pthread_timedjoin_np(simulator_state.axon_dsp.int_thread_handle, NULL, &ts);
  return 0;
}

void exit_simulator_axon_dsp() {
  simulator_state.axon_dsp.terminate_thread = true;
  int wait_result =  pthread_join(simulator_state.axon_dsp.hw_thread_handle, NULL);
  if(wait_result!=0){
    printf("AXON DSP HW Thread did not exit cleanly! \n");
  }

  // for the interrupt thread, release a semaphore which will then cause it to exit cleanly
  sem_post(&simulator_state.axon_dsp.terminate_int_thread_semaphore);
  wait_result = pthread_join(simulator_state.axon_dsp.int_thread_handle, NULL);
  // and if that is successful close the interrupt thread handle
  if(wait_result!=0){
     printf("AXON DSP HW Interrupt Thread did not exit cleanly! \n");
  }
  pthread_mutex_destroy(&simulator_state.axon_dsp.registers_mutex);
  sem_destroy(&simulator_state.axon_dsp.terminate_int_thread_semaphore);
  sem_destroy(&simulator_state.axon_dsp.semaphore);
}

/* ---- end of axon dsp---- */

/*
 * Axon NN write to register (executes in the application thread only)
 */
void axon_nn_simulator_write_reg(volatile uint64_t *addr, uint64_t value) {

  // grab mutex by calling wait()
  simulator_state.axon_nn.mutex_wait_result = pthread_mutex_lock(&simulator_state.axon_nn.registers_mutex);

  // catch failed wait()
  if (simulator_state.axon_nn.mutex_wait_result != 0) {
    printf("AXON NN Mutex wait failed..");
    system("PAUSE");
  }

  switch(axon_nn_simulator_write_reg_prim(addr, value)) {
    case 0: break;
    case 1: simulator_state.axon_nn.action_reg_updated = true; break;
    case 2: sem_post(&simulator_state.axon_nn.semaphore); break;
  }
  pthread_mutex_unlock(&simulator_state.axon_nn.registers_mutex);
}

/**
 * Axon NN read from register (executes in the application thread only)
 */
uint64_t axon_nn_simulator_read_reg(volatile uint64_t *addr) {

  // grab mutex by calling wait()
  simulator_state.axon_nn.mutex_wait_result = pthread_mutex_lock(&simulator_state.axon_nn.registers_mutex);

  // catch failed wait()
  if (simulator_state.axon_nn.mutex_wait_result != 0) {
    printf("AXON NN Mutex wait failed..");
    system("PAUSE");
  }

  // read value from application register set
  uint64_t return_value = axon_nn_simulator_read_reg_prim(addr);
  pthread_mutex_unlock(&simulator_state.axon_nn.registers_mutex);
  return return_value;

}

bool axon_nn_simualtor_int_pending() {
  if (0 != pthread_mutex_lock(&simulator_state.axon_nn.registers_mutex)) {
    printf("AXON Mutex wait failed..");
    system("PAUSE");
  }
  bool result = axon_nn_simualtor_int_pending_prim();
  pthread_mutex_unlock(&simulator_state.axon_nn.registers_mutex);
  return result;
}

void* axon_nn_int_thread(void* data) {
  bool int_fired = false;
  while (1) {
    if(int_fired || (sem_trywait(&simulator_state.axon_nn.semaphore)==0)){
      int_fired = true;
      // allow for the int firing while disabled, then the int is cleared. Need to suppress the ISR in that case.
      if (axon_simulator_ints_enabled) {
        if (axon_nn_simualtor_int_pending()) {
            // call the handler
            host_irq_handler(NULL);
        }
        // only clear  int_fired if the int bits were cleared.
        int_fired = axon_nn_simualtor_int_pending();
      }
    }
    if(sem_trywait(&simulator_state.axon_nn.terminate_int_thread_semaphore)==0){
      break;
    }
  }
  return 0;
}


// Axon NN hardware thread
void* axon_nn_hw_thread(void *data) {
  // // register polling loop
  while (1) {

    if (simulator_state.axon_nn.terminate_thread) {
      break;
    }

    // wait for driver to write something of interest...
    if (simulator_state.axon_nn.action_reg_updated == 0) {
      continue; // ...nothing yet, skip below and go back to the beginning of while loop
    }

    // action is necessary - grab mutex by calling wait()
    simulator_state.axon_nn.mutex_wait_result = pthread_mutex_lock(&simulator_state.axon_nn.registers_mutex);

    // catch failed wait()
    if (simulator_state.axon_nn.mutex_wait_result != 0) {
      printf("AXON NN Mutex wait failed..");
      system("PAUSE");
    }

    // clear out action indicator
    simulator_state.axon_nn.action_reg_updated = 0;

    pthread_mutex_unlock(&simulator_state.axon_nn.registers_mutex);
    /* check interrupts*/
    int nn_simulator_err, nn_o_wdog_cmd, nn_o_wdog_finish;
    if (axon_nn_simulator_process_action_request(&nn_simulator_err, &nn_o_cycles, &nn_o_wdog_cmd, &nn_o_wdog_finish)) {
      sem_post(&simulator_state.axon_nn.semaphore);
    }

  }
  return 0;
}

static void* start_simulator_axon_nn() {

  simulator_state.axon_nn.terminate_thread = false;

  // create mutex
  int ret = pthread_mutex_init(&simulator_state.axon_nn.registers_mutex,NULL);
  if (ret!=0) {
    printf("AXON NN CreateMutex error: %d\n", ret);
    return 0;
  }

  // create semaphore
  ret = sem_init(&simulator_state.axon_nn.semaphore,0,0);
  if (ret!=0) {
    printf("AXON NN CreateSemaphore error: %d\n",ret);
    system("PAUSE");
  }

  // create interrupt thread semaphore
   ret = sem_init(&simulator_state.axon_nn.terminate_int_thread_semaphore, 0, 0);
  if (ret!=0) {
    printf("AXON NN CreateSemaphore interrupt thread terminate error: %d\n", ret);
    system("PAUSE");
  }

  // initialize register values, returns the base address
  void * axon_base_addr = axon_nn_initialize_registers();
  // create threads for axon nn hardware and the axon nn interrupt
  /**
   * @FIXME! Check axon cmodel stack size requirements ~64kB
   * EC  - find where stack size is being set/overwritten in exe
   */
  ret = pthread_create(&simulator_state.axon_nn.hw_thread_handle, 0, axon_nn_hw_thread, NULL);
  if (ret!= 0) {
    printf("AXON NN HW Thread creation failed. Exiting program \n");
    fflush(stdout);
    system("PAUSE");
  }

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec +=1; //wait for 1 second
  pthread_timedjoin_np(simulator_state.axon_nn.hw_thread_handle, NULL, &ts);
  ret = pthread_create(&simulator_state.axon_nn.int_thread_handle, 0, axon_nn_int_thread, NULL);
  if (ret!= 0) {
    printf("AXON NN INT Thread creation failed. Exiting program \n");
    fflush(stdout);
    system("PAUSE");
  }
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec +=1; //wait for 1 second
  pthread_timedjoin_np(simulator_state.axon_nn.int_thread_handle, NULL, &ts);
  return axon_base_addr;
}

void exit_simulator_axon_nn() {
  simulator_state.axon_nn.terminate_thread = true;
  int wait_result =  pthread_join(simulator_state.axon_nn.hw_thread_handle, NULL);
  if(wait_result!=0){
    printf("AXON NN HW Thread did not exit cleanly! \n");
  }

  // for the interrupt thread, release a semaphore which will then cause it to exit cleanly
  sem_post(&simulator_state.axon_nn.terminate_int_thread_semaphore);
  wait_result = pthread_join(simulator_state.axon_nn.int_thread_handle, NULL);
  // and if that is successful close the interrupt thread handle
  if(wait_result!=0){
     printf("AXON NN HW Interrupt Thread did not exit cleanly! \n");
  }
  pthread_mutex_destroy(&simulator_state.axon_nn.registers_mutex);
  sem_destroy(&simulator_state.axon_nn.terminate_int_thread_semaphore);
  sem_destroy(&simulator_state.axon_nn.semaphore);
}

/* ---- end of axon_nn ---- */

/*
* Starts the nn & dsp compute unit threads, returns the register base address.
*/
void *start_simulator() {
   static_assert(sizeof(void *) == sizeof(uint64_t), "platform architecture setting mismatch!");
   printf(" platform architecture %d-bit\n", (int)sizeof(void *)*8);
   start_simulator_axon_dsp();
   return start_simulator_axon_nn();
}

void exit_simulator() {
  exit_simulator_axon_dsp();
  exit_simulator_axon_nn();
}

uint32_t nrf_axon_platform_get_ticks(){
  struct timespec ts;
  uint32_t ticks=0;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  ticks = ts.tv_nsec / 1000000;
  ticks += ts.tv_sec * 1000;
  return ticks;
}

int fopen_s(FILE** f, const char* name, const char* mode) {
  int ret = 0;
  assert(f);
  *f = fopen(name, mode);
  if (*f==NULL)
      ret = -1;
  return ret;
}

int fprintf_s(FILE * stream,const char * format, ...) {
  va_list args;
  va_start(args, format);
  char tempstring[512];
  vsnprintf(tempstring, sizeof(tempstring),format, args);
  va_end(args);
  return fprintf(stream,"%s", tempstring);
}

int fscanf_s(FILE * stream,const char * format, ...) {
  va_list args;
  va_start(args, format);
  char tempstring[512];
  int result = vfscanf(stream,format,args);
  va_end(args);
  return result;
}

size_t fread_s(void * buffer_ptr, size_t buffer_size, size_t size, size_t n, FILE * stream) {
  return fread(buffer_ptr, size, n, stream);
}

char *strcpy_s(char* dest, size_t size, char const* src) {
  size_t source_size = strlen(src) + 1;

  if (source_size<=size) {
    strcpy(dest,src);
  } else {
    memcpy(dest,src,size-1);
    dest[size-1] = 0; // terminate it.
  }
  return dest;
}

int axon_simulator_run_test_files(
  char* input_file_path,
  char* output_file_path,
  char* input_file_ext,
  char* output_file_head_str,
  uint32_t buffer_size,
  int (*callback_function)(char* input_file_name, char* output_file_name, int8_t* buffer, uint32_t buffer_size)) {

  #define STR_BUF_SIZE 512
  char fullPath_input[STR_BUF_SIZE]; // Ensure this is large enough to hold the full path
  char fullPath_output[STR_BUF_SIZE]; // Ensure this is large enough to hold the full path

  struct dirent *entry;
  if((input_file_path[strnlen(input_file_path, STR_BUF_SIZE)-1] != '\\') && (input_file_path[strnlen(input_file_path, STR_BUF_SIZE)-1] != '/')) {
    snprintf(fullPath_input, sizeof(fullPath_input), "%s/", input_file_path);
  } else {
    snprintf(fullPath_input, sizeof(fullPath_input), "%s", input_file_path);
  }

  DIR *dp = opendir(fullPath_input);

  if (dp == NULL) {
      nrf_axon_platform_printf("error: opendir %s\n", input_file_path);
      return -1;
  }

  // make sure this is 32bit aligned in case it gets down-cast from int8*
  int32_t* buffer = malloc(buffer_size);
  if (NULL==buffer) {
    return -2;
  }

  uint32_t input_files_processed = 0;
  while ((entry = readdir(dp))) {
      if (strstr(entry->d_name, input_file_ext)) {
          nrf_axon_platform_printf("Processing: %s\n", entry->d_name);
          if((input_file_path[strnlen(input_file_path, STR_BUF_SIZE)-1] != '\\') && (input_file_path[strnlen(input_file_path, STR_BUF_SIZE)-1] != '/')) {
            snprintf(fullPath_input, sizeof(fullPath_input), "%s/%s", input_file_path, entry->d_name);
          } else {
            snprintf(fullPath_input, sizeof(fullPath_input), "%s%s", input_file_path, entry->d_name);
          }

          DIR *outp = opendir(output_file_path);
          if(NULL == outp) {
            nrf_axon_platform_printf("%s doesn't exist, create it!\n", output_file_path);
            mkdir(output_file_path, 0775);
          } else {
            closedir(outp);
          }
          if((output_file_path[strnlen(output_file_path, STR_BUF_SIZE)-1] != '\\') && (output_file_path[strnlen(output_file_path, STR_BUF_SIZE)-1] != '/')) {
            snprintf(fullPath_output, sizeof(fullPath_output), "%s/%s%s", output_file_path, output_file_head_str, entry->d_name);
          } else {
            snprintf(fullPath_output, sizeof(fullPath_output), "%s%s%s", output_file_path, output_file_head_str, entry->d_name);
          }

          callback_function(fullPath_input, fullPath_output, (int8_t*)buffer, buffer_size);
          input_files_processed++;
      }
  }
  nrf_axon_platform_printf("%d files processed\n", input_files_processed);

  closedir(dp);

  free(buffer);

  axon_simulator_print_saturation_statistics();

  return 0;
}
