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
#include <assert.h>
#include <zephyr/kernel.h>
#include <zephyr/irq.h>
#include <zephyr/sys_clock.h>
#include <zephyr/sys/printk.h>
#include <zephyr/devicetree.h>

#define USE_DTS_GPIO 1

#if USE_DTS_GPIO
# include <zephyr/drivers/gpio.h>
#else
# include <hal/nrf_gpio.h>
#endif

#include <zephyr/logging/log.h>
#include "nrf_axon_driver.h"
#include "nrf_axon_platform_interface.h"
#include "nrf_axon_platform.h"

#include <nrfx.h>

#define AXON_NN_BASE_ADDR ((void *) DT_REG_ADDR(AXONS))
#define AXON_DSP_BASE_ADDR AXON_NN_BASE_ADDR

#define AXONS DT_NODELABEL(axons)
#define AXONS_IRQ_NO DT_IRQN(AXONS)
#define AXONS_IRQ_PRIORITY DT_IRQ(AXONS, priority)

#if USE_DTS_GPIO
/* The devicetree node identifier for the "axonprofilinggpio0" alias. */
# define AXON_PROFILING_GPIO_NODE DT_ALIAS(axonprofilinggpio0)
# if DT_NODE_EXISTS(AXON_PROFILING_GPIO_NODE)
static const struct gpio_dt_spec axon_profiling_gpio = GPIO_DT_SPEC_GET(AXON_PROFILING_GPIO_NODE, gpios);
# endif
#else
# define AXONS_PROFILING_GPIO_PIN NRF_GPIO_PIN_MAP(1,0)
#endif

void nrf_axon_platform_set_profiling_gpio()
{
#if USE_DTS_GPIO 
# if DT_NODE_EXISTS(AXON_PROFILING_GPIO_NODE)
	gpio_pin_configure_dt(&axon_profiling_gpio, GPIO_OUTPUT_ACTIVE);
  gpio_pin_set_dt(&axon_profiling_gpio, 1);
# endif
#else
  nrf_gpio_cfg(AXONS_PROFILING_GPIO_PIN,NRF_GPIO_PIN_DIR_OUTPUT,NRF_GPIO_PIN_INPUT_CONNECT,NRF_GPIO_PIN_PULLUP,NRF_GPIO_PIN_H0H1,NRF_GPIO_PIN_NOSENSE);
  nrf_gpio_cfg_output(AXONS_PROFILING_GPIO_PIN);
  nrf_gpio_pin_set(AXONS_PROFILING_GPIO_PIN);
#endif
}

void nrf_axon_platform_clear_profiling_gpio()
{
#if USE_DTS_GPIO
# if DT_NODE_EXISTS(AXON_PROFILING_GPIO_NODE)
	gpio_pin_configure_dt(&axon_profiling_gpio, GPIO_OUTPUT_ACTIVE);
  gpio_pin_set_dt(&axon_profiling_gpio, 0);
# endif
#else
  nrf_gpio_cfg_output(AXONS_PROFILING_GPIO_PIN);
  nrf_gpio_pin_clear(AXONS_PROFILING_GPIO_PIN);
#endif
}

#define AXONS_DRIVER_STACK_SIZE 2000 
#define AXONS_DRIVER_PRIORITY 10

// semaphore to signal the driver thread. Set the maximum value > 1 in case interrupts get stacked up.
static K_SEM_DEFINE(axons_isr_sem, 0, 10);

// semaphore to signal the user thread on 
static K_SEM_DEFINE(axons_user_event_sem, 0, 1);

void nrf_axon_platform_wait_for_user_event() {
    k_sem_take(&axons_user_event_sem, K_FOREVER);
}
void nrf_axon_platform_generate_user_event(){
    k_sem_give(&axons_user_event_sem);
}

#if ((NRF_AXON_INTERLAYER_BUFFER_SIZE) > 0) 
__attribute__ ((section (NRF_AXON_INTERLAYER_BUFFER_MEMREGION)))
uint32_t nrf_axon_interlayer_buffer[NRF_AXON_INTERLAYER_BUFFER_SIZE/sizeof(uint32_t)];
#endif

#if ((NRF_AXON_PSUM_BUFFER_SIZE) > 0)
__attribute__ ((section (NRF_AXON_INTERLAYER_BUFFER_MEMREGION)))
uint32_t nrf_axon_psum_buffer[NRF_AXON_PSUM_BUFFER_SIZE/sizeof(uint32_t)];
#endif

#if 0
int abs(int in) {
  return in >= 0 ? in : -in;
}
#endif

void delay_us(uint32_t delay) {
	k_busy_wait(delay);
}

// #define MY_LOG_LEVEL LOG_LEVEL_NONE
// #define MY_LOG_LEVEL LOG_LEVEL_DBG
#define MY_LOG_LEVEL LOG_LEVEL_INF

LOG_MODULE_REGISTER(mlss, MY_LOG_LEVEL);
// LOG_MODULE_DECLARE();

void nrf_axon_platform_printf(const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  // char tempstring[512]; //increased the buffer length from 255 to 512 to accommodate larger strings
  // vsnprintf(tempstring, sizeof(tempstring), fmt, args);
  vprintk(fmt, args);
  va_end(args);
  //LOG_INF("%s", tempstring);
}

/**
 * Console logging function implemented by host.
 */
void nrf_axon_platform_log(char *msg) 
{
  LOG_INF("%s", msg);
}

uint32_t nrf_axon_platform_get_clk_hz() 
{
  return sys_clock_hw_cycles_per_sec();
}
uint32_t nrf_axon_platform_get_ticks() {
  return k_cycle_get_32();
}

static void enable_axon_interrupt() {
	irq_enable(AXONS_IRQ_NO);
}

static void disable_axon_interrupt() {
  irq_disable(AXONS_IRQ_NO);
}

uint32_t nrf_axon_platform_disable_interrupts() {
  return irq_lock();
}

void nrf_axon_platform_restore_interrupts(uint32_t restore_value) {
  irq_unlock(restore_value);
}

/*
 * Registered ISR for axonpro interrupt
 * In our simple application we handle IRQs directly
 */
static void axon_platform_irq_handler(void * data) {
  /* Axon ISR will invoke nrf_axon_platform_generate_driver_event() if further processing is needed. */
  nrf_axon_handle_interrupt(); 
}

/**
 * host function for enabling axonnn by powering it up and providing a clock.
 */
// FIXME!!! THERE SHOULD BE SOME SYSTEM LEVEL VOTING!!!!
#if defined RRAMC_POWER_CONFIG_POF_Pos
static volatile uint32_t *rramc_lowpowerconfig_ptr = &NRF_RRAMC->POWER.LOWPOWERCONFIG;
static uint32_t LOWPOWERCONFIG_RESTOREVALUE;
static volatile uint32_t *rramc_powerstandbyconfig_ptr = &NRF_RRAMC->POWER.RESERVED;
static uint32_t  STANDBYCONFIG_RESTOREVALUE;
#endif
#define AXON_REG_ENABLE_OFFSET 0x400
static void axon_enable() {
# if !defined(AXONS_ENABLE_EN_Pos)
  *(uint32_t*)((int8_t*)AXON_NN_BASE_ADDR+AXON_REG_ENABLE_OFFSET) |= 1;
# else
  /**
   * @fixme!! will this be in the public mdk? if so, nrf.h can be included and this code used.
   */
  NRF_AXONS_Type *nrf_axons = (NRF_AXONS_Type *)((int8_t*)AXON_NN_BASE_ADDR+AXON_REG_ENABLE_OFFSET);
  nrf_axons->ENABLE |= (AXONS_ENABLE_EN_Enabled << AXONS_ENABLE_EN_Pos);
# endif
#if defined RRAMC_POWER_LOWPOWERCONFIG_MODE_Pos
  uint32_t tmp = *rramc_lowpowerconfig_ptr;
  LOWPOWERCONFIG_RESTOREVALUE = tmp;
  tmp &= ~RRAMC_POWER_LOWPOWERCONFIG_MODE_Msk;
  tmp |= (RRAMC_POWER_LOWPOWERCONFIG_MODE_Standby    << RRAMC_POWER_LOWPOWERCONFIG_MODE_Pos);
  *rramc_lowpowerconfig_ptr = tmp;

  tmp = *rramc_powerstandbyconfig_ptr;
  STANDBYCONFIG_RESTOREVALUE = tmp;
  *rramc_powerstandbyconfig_ptr = 3;
#else
  static_assert(0, "MISSING RRAMC!!!");
#endif
}

/**
 * host function for disabling axonnn by powering it down and removing the clock.
 */
static void axon_disable(){

#if !defined(AXONS_ENABLE_EN_Pos)
  *(uint32_t*)((int8_t*)AXON_NN_BASE_ADDR+AXON_REG_ENABLE_OFFSET) &= ~1;
#else
  /**
   * @fixme!! will this be in the public mdk? if so, nrf.h can be included and this code used.
   */
  NRF_AXONS_Type *nrf_axons = (NRF_AXONS_Type *)AXON_NN_BASE_ADDR;
  nrf_axons->ENABLE &= ~(AXONS_ENABLE_EN_Enabled << AXONS_ENABLE_EN_Pos);
#endif
#if defined RRAMC_POWER_LOWPOWERCONFIG_MODE_Pos
  *rramc_lowpowerconfig_ptr = LOWPOWERCONFIG_RESTOREVALUE;
  *rramc_powerstandbyconfig_ptr = STANDBYCONFIG_RESTOREVALUE;

#endif
#if defined POWER_TASKS_LOWPWR_TASKS_LOWPWR_Pos
  NRF_POWER_S->TASKS_LOWPWR |= (POWER_TASKS_LOWPWR_TASKS_LOWPWR_Trigger << POWER_TASKS_LOWPWR_TASKS_LOWPWR_Pos);
#endif
}

/**
 * Thread that is signaled by axons interrupt
*/
static void axons_driver_thread(void *unused1, void *unused2, void *unused3) {
  while (1) {
    k_sem_take(&axons_isr_sem, K_FOREVER);
    nrf_axon_process_driver_event();
  }
}
/**
 * platform function to generate an event that will cause nrf_axon_process_event() to be invoked.
 */
void nrf_axon_platform_generate_driver_event() {
  k_sem_give(&axons_isr_sem);
}

/*
* Declares the thread at compile time with no start delay.
*/
K_THREAD_DEFINE(axons_driver_tid, AXONS_DRIVER_STACK_SIZE,
                axons_driver_thread, NULL, NULL, NULL,
                AXONS_DRIVER_PRIORITY, 0, 0);

// semaphore to reserve access to axon hardware. Used to regulate sync and async modes.
static K_SEM_DEFINE(axons_reserve_sem, 1, 1);
static volatile k_tid_t axons_owner_thread_id = NULL; // k_tid_t is a pointer

/**
 * Increments the power vote. If this is the 1st 1, axon is powered on.
 * Returns the number of outstanding votes.
*/
static volatile int axon_power_vote_cnt = 0;
static int vote_for_power()
{
  uint32_t int_state = nrf_axon_platform_disable_interrupts();
  bool do_enable = 0==axon_power_vote_cnt++;
  nrf_axon_platform_restore_interrupts(int_state);
  if (do_enable) {
      axon_enable();
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
  uint32_t int_state = nrf_axon_platform_disable_interrupts();
  bool do_disable = (0==--axon_power_vote_cnt);
  if (0 > axon_power_vote_cnt)
    axon_power_vote_cnt = 0;
  nrf_axon_platform_restore_interrupts(int_state);
  if (do_disable) {
    nrf_axon_driver_power_off();
    axon_disable();
  }
  return axon_power_vote_cnt;
}

/**
 * Driver is asynchronous, so no waiting on the semaphore.
 */
bool nrf_axon_platform_reserve_for_driver() {
  /* vote for power here even if the reservation failed. The pending power vote will prevent
     Axon from being powered down when the user releases Axon.*/
  vote_for_power(); 

  if (0 == k_sem_take(&axons_reserve_sem, K_NO_WAIT)) {
    /* driver will never reserve from its own thread, always from a user thread. */
    axons_owner_thread_id = axons_driver_tid;
    return true;
  }
  return false;
}

/**
 * to use the hardware in synchronous mode, must reserve for
 * exclusive use.
 * This function will be called by the driver during asynchronous inference when intrinsics are invoked by the CPU. When this happens,
 * the driver will already own the reservation, and the call will always happen in the driver thread, so the driver appears as a user at 
 * this point and the function will short-circuit.
 */
bool nrf_axon_platform_reserve_for_user() {
  
  /* check if already own the reservation */
  if(k_current_get()==axons_owner_thread_id) {
    return true; // already own it, short-circuit.
  }
  /* if the driver thread ever requests to reserve for user it is because intrinsics are part of the command buffer
     and so axon is available*/
  if ((NULL != axons_owner_thread_id) && (k_current_get() == axons_driver_tid)) {
    return true;
  }
  /* don't already own it, so need to wait. */
  vote_for_power(); // vote for power now to avoid a glitch when it is freed.

  k_sem_take(&axons_reserve_sem, K_FOREVER);
  axons_owner_thread_id = k_current_get();
  return true;
}

/**
 * this only gets called from user threads. If a user is freeing the driver, the driver has priority to
 * get it. 
 */
void nrf_axon_platform_free_reservation_from_user() {
  /* */
  vote_against_power();

  if (nrf_axon_queue_not_empty()) {
    // driver needs the axon hardware, so don't free the sem, just start the hardware. It will have its own power vote pending.
    axons_owner_thread_id = axons_driver_tid;
    nrf_axon_start_queue_processing();
    return;
  }
  axons_owner_thread_id = NULL;
  k_sem_give(&axons_reserve_sem);
}

void nrf_axon_platform_free_reservation_from_driver() {
  axons_owner_thread_id = NULL;
  vote_against_power(); // power down axon
  k_sem_give(&axons_reserve_sem);
}

nrf_axon_result_e nrf_axon_platform_init() {
  LOG_DBG("AXONS_BASE_ADDR: 0x%p", AXON_NN_BASE_ADDR);
  nrf_axon_result_e result;
  axon_enable();
  nrf_axon_platform_printf("axon on");
  if (NRF_AXON_RESULT_SUCCESS != (result=nrf_axon_driver_init(AXON_NN_BASE_ADDR))) {
    return result;
  }
  
  IRQ_CONNECT(AXONS_IRQ_NO, AXONS_IRQ_PRIORITY, axon_platform_irq_handler, 0, 0);
  enable_axon_interrupt();
  
  axon_disable();

  return NRF_AXON_RESULT_SUCCESS;
}


void nrf_axon_platform_close() {
  axon_power_vote_cnt=0;
  axon_disable();
}
