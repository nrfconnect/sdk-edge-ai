/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <zephyr/kernel.h>
#include <zephyr/irq.h>
#include <zephyr/sys_clock.h>
#include <zephyr/sys/onoff.h>
#include <zephyr/sys/printk.h>
#include <zephyr/devicetree.h>

#include <zephyr/drivers/gpio.h>

#include <zephyr/logging/log.h>
#include "drivers/axon/nrf_axon_driver.h"
#include "drivers/axon/nrf_axon_platform_interface.h"
#include "axon/nrf_axon_platform.h"

#include <mdk/nrf.h>

#define NRF_AXON_NVM_RRAM 1
#define NRF_AXON_NVM_MRAM 2
#define NRF_AXON_NVM_MRAM_9251 3
#if defined(CONFIG_SOC_NRF7120)
# define NRF_AXON_NVM_TYPE NRF_AXON_NVM_MRAM
# define NRF_AXON_IN_MDK 0
#elif defined(CONFIG_SOC_NRF9251)
# define NRF_AXON_NVM_TYPE NRF_AXON_NVM_MRAM_9251
# define NRF_AXON_IN_MDK 0
# include <ironside_zephyr/se/uicr_periphconf.h>
#elif defined(CONFIG_SOC_NRF54LM20B)
# define NRF_AXON_NVM_TYPE NRF_AXON_NVM_RRAM
/**
 * @FIXME!!! zephyr/soc/nordic/nrf54l/Kconfig file specifies
 * config SOC_NRF54LM20B_CPUAPP
 *  select SOC_NRF54L_CPUAPP_COMMON
 *  select SOC_COMPATIBLE_NRF54LM20B
 *  select SOC_COMPATIBLE_NRF54LM20B_CPUAPP
 *  select ARMV8_M_DSP
 *  select CPU_HAS_ARM_MPU
 *  select CPU_HAS_ARM_SAU
 *  select CPU_HAS_FPU
 *  select HAS_SWO
 *
 * This results in both
 * NRF54LM20A_XXAA AND NRF54LM20B_XXAA being defined.
 * in modules\hal\nordic\nrfx\bsp\stable\mdk\nrf.h, defining NRF54LM20A_XXAA
 * causes the 20A MDK files to be included, not the 20B, so
 * we don't see axon definitions.
 *
 * When this is resolved, set NRF_AXON_IN_MDK to 1.
 */
# define NRF_AXON_IN_MDK 0 /* should be 1! */
#endif

#if (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_RRAM)
# include <nrf_sys_event.h>
static int sys_event_handle;
#elif (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_MRAM)
static uint32_t mramc_autopowerdown_restore;
#elif (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_MRAM_9251)
# include <nrfs_mram.h>

static K_SEM_DEFINE(nrf_axon_mram_latency_sem, 0, 1);
static int nrf_axon_mram_latency_result;

static void nrf_axon_mram_latency_evt_handler(nrfs_mram_latency_evt_t const *p_evt,
					     void *context)
{
	nrf_axon_mram_latency_result =
		(p_evt->type == NRFS_MRAM_LATENCY_REQ_APPLIED) ? 0 : -EIO;

	k_sem_give((struct k_sem *)context);
}

static int nrf_axon_mram_set_latency_sync(mram_latency_request_t latency_request)
{
	nrfs_err_t err;

	err = nrfs_mram_set_latency(latency_request, (void *)&nrf_axon_mram_latency_sem);
	if (err != NRFS_SUCCESS) {
		return -EIO;
	}

	/* MRAM_LATENCY_ALLOWED does not produce a callback event. */
	if (latency_request == MRAM_LATENCY_ALLOWED) {
		return 0;
	}

	if (k_sem_take(&nrf_axon_mram_latency_sem, K_MSEC(1000)) != 0) {
		return -ETIMEDOUT;
	}

	return nrf_axon_mram_latency_result;
}
#endif

#define AXON_DT_NODELABEL DT_NODELABEL(axon)

#define AXON_BASE_ADDR ((void *) DT_REG_ADDR(AXON_DT_NODELABEL))

#define AXON_IRQ_NO DT_IRQN(AXON_DT_NODELABEL)
#define AXON_IRQ_PRIORITY DT_IRQ(AXON_DT_NODELABEL, priority)

static struct onoff_manager power_mgr;
static struct k_work axon_driver_work;

/* The devicetree node identifier for the "axonprofilinggpio0" alias. */
#define AXON_PROFILING_GPIO_NODE DT_ALIAS(axonprofilinggpio0)
#if DT_NODE_EXISTS(AXON_PROFILING_GPIO_NODE)
static const struct gpio_dt_spec axon_profiling_gpio =
	GPIO_DT_SPEC_GET(AXON_PROFILING_GPIO_NODE, gpios);
#endif

void nrf_axon_platform_set_profiling_gpio(void)
{
#if DT_NODE_EXISTS(AXON_PROFILING_GPIO_NODE)
	gpio_pin_configure_dt(&axon_profiling_gpio, GPIO_OUTPUT_ACTIVE);
	gpio_pin_set_dt(&axon_profiling_gpio, 1);
#endif
}

void nrf_axon_platform_clear_profiling_gpio(void)
{
#if DT_NODE_EXISTS(AXON_PROFILING_GPIO_NODE)
	gpio_pin_configure_dt(&axon_profiling_gpio, GPIO_OUTPUT_ACTIVE);
	gpio_pin_set_dt(&axon_profiling_gpio, 0);
#endif
}

/* semaphore to signal the user thread on */
static K_SEM_DEFINE(axon_user_event_sem, 0, 1);

void nrf_axon_platform_wait_for_user_event(void)
{
	k_sem_take(&axon_user_event_sem, K_FOREVER);
}
void nrf_axon_platform_generate_user_event(void)
{
	k_sem_give(&axon_user_event_sem);
}

#if ((NRF_AXON_INTERLAYER_BUFFER_SIZE) > 0)
__attribute__ ((section (NRF_AXON_INTERLAYER_BUFFER_MEMREGION)))
uint32_t nrf_axon_interlayer_buffer[NRF_AXON_INTERLAYER_BUFFER_SIZE/sizeof(uint32_t)];
#endif

#if ((NRF_AXON_PSUM_BUFFER_SIZE) > 0)
__attribute__ ((section (NRF_AXON_INTERLAYER_BUFFER_MEMREGION)))
uint32_t nrf_axon_psum_buffer[NRF_AXON_PSUM_BUFFER_SIZE/sizeof(uint32_t)];
#endif

void delay_us(uint32_t delay)
{
	k_busy_wait(delay);
}

// #define MY_LOG_LEVEL LOG_LEVEL_NONE
// #define MY_LOG_LEVEL LOG_LEVEL_DBG
#define MY_LOG_LEVEL LOG_LEVEL_INF

LOG_MODULE_REGISTER(mlss, MY_LOG_LEVEL);
// LOG_MODULE_DECLARE();

void nrf_axon_platform_printf(const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	vprintk(fmt, args);
	va_end(args);
}

/**
 * Console logging function implemented by host.
 */
void nrf_axon_platform_log(char *msg)
{
	LOG_INF("%s", msg);
}

uint32_t nrf_axon_platform_get_clk_hz(void)
{
	return sys_clock_hw_cycles_per_sec();
}
uint32_t nrf_axon_platform_get_ticks(void)
{
	return k_cycle_get_32();
}

static void enable_axon_interrupt(void)
{
	irq_enable(AXON_IRQ_NO);
}

#if 0
// not currently used.
static void disable_axon_interrupt(void)
{
	irq_disable(AXON_IRQ_NO);
}
#endif

uint32_t nrf_axon_platform_disable_interrupts(void)
{
	return irq_lock();
}

void nrf_axon_platform_restore_interrupts(uint32_t restore_value)
{
	irq_unlock(restore_value);
}

/*
 * Registered ISR for axonpro interrupt
 * In our simple application we handle IRQs directly
 */
static void axon_platform_irq_handler(void *data)
{
	/**
	 * Axon ISR will invoke nrf_axon_platform_generate_driver_event()
	 * if further processing is needed.
	 */
	nrf_axon_handle_interrupt();
}

/**
 * host function for enabling axonnn by powering it up and providing a clock.
 */
#define AXON_REG_ENABLE_OFFSET 0x400
static void axon_enable(void)
{
# if !NRF_AXON_IN_MDK
	*(uint32_t *)((int8_t *)AXON_BASE_ADDR+AXON_REG_ENABLE_OFFSET) |= 1;

# else
	NRF_AXONS_Type *nrf_axons = (NRF_AXONS_Type *)((int8_t *)AXON_BASE_ADDR);

	nrf_axons->ENABLE |= (AXONS_ENABLE_EN_Enabled << AXONS_ENABLE_EN_Pos);
# endif

	/* Register an event starting now to make RRAM stay in standby mode. */
#if (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_RRAM)
	sys_event_handle = nrf_sys_event_register(0, true);
#endif

#if (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_MRAM)
	mramc_autopowerdown_restore = NRF_MRAMC->POWER.AUTOPOWERDOWN;
	NRF_MRAMC->POWER.AUTOPOWERDOWN = ((MRAMC_POWER_AUTOPOWERDOWN_ENABLE_Msk &
		(MRAMC_POWER_AUTOPOWERDOWN_ENABLE_Max << MRAMC_POWER_AUTOPOWERDOWN_ENABLE_Pos)) |
		(MRAMC_POWER_AUTOPOWERDOWN_TIMEOUTVALUE_Msk &
		(MRAMC_POWER_AUTOPOWERDOWN_TIMEOUTVALUE_Max <<
		 MRAMC_POWER_AUTOPOWERDOWN_TIMEOUTVALUE_Pos)));
#elif (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_MRAM_9251)
	int rc = nrf_axon_mram_set_latency_sync(MRAM_LATENCY_NOT_ALLOWED);

	if (rc != 0) {
		LOG_ERR("MRAM latency NOT_ALLOWED failed: %d", rc);
	}
#endif
}

/**
 * host function for disabling axonnn by powering it down and removing the clock.
 */
static void axon_disable(void)
{
#if !NRF_AXON_IN_MDK
	*(uint32_t *)((int8_t *)AXON_BASE_ADDR+AXON_REG_ENABLE_OFFSET) &= ~1;

#else
	NRF_AXONS_Type *nrf_axons = (NRF_AXONS_Type *)AXON_BASE_ADDR;

	nrf_axons->ENABLE &= ~(AXONS_ENABLE_EN_Enabled << AXONS_ENABLE_EN_Pos);
#endif

	/* Deregister the event so that RRAM can be powered off. */
#if (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_RRAM)
	nrf_sys_event_unregister(sys_event_handle, false);
#endif
#if (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_MRAM)
	NRF_MRAMC->POWER.AUTOPOWERDOWN = mramc_autopowerdown_restore;
#elif (NRF_AXON_NVM_TYPE) == (NRF_AXON_NVM_MRAM_9251)
	int rc = nrf_axon_mram_set_latency_sync(MRAM_LATENCY_ALLOWED);

	if (rc != 0) {
		LOG_ERR("MRAM latency ALLOWED failed: %d", rc);
	}
#endif
}

static void axon_driver_work_handler(struct k_work *work)
{
	ARG_UNUSED(work);

	nrf_axon_process_driver_event();
}

void nrf_axon_platform_generate_driver_event(void)
{
	k_work_submit(&axon_driver_work);
}

#define AXON_DRIVER_THREAD_ID k_work_queue_thread_get(&k_sys_work_q)

/* semaphore to reserve access to axon hardware. Used to regulate sync and async modes. */
static K_SEM_DEFINE(axon_reserve_sem, 1, 1);
static volatile k_tid_t axon_owner_thread_id;

static void axon_power_on(struct onoff_manager *mgr, onoff_notify_fn notify)
{
	axon_enable();
	nrf_axon_driver_power_on();

	notify(mgr, 0);
}

static void axon_power_off(struct onoff_manager *mgr, onoff_notify_fn notify)
{
	nrf_axon_driver_power_off();
	axon_disable();

	notify(mgr, 0);
}

/* Requests that Axon is powered on. It will be kept powered on as long as
 * there is at least one such request active.
 */
static void axon_power_request(void)
{
	struct onoff_client cli;
	int rc;
	int result;

	sys_notify_init_spinwait(&cli.notify);
	rc = onoff_request(&power_mgr, &cli);
	__ASSERT_NO_MSG(rc == 0);

	do {
		rc = sys_notify_fetch_result(&cli.notify, &result);
	} while (rc == -EAGAIN);

	__ASSERT_NO_MSG(result == 0);
}

/* Releases Axon power requested previously with axon_power_request(). */
static void axon_power_release(void)
{
	onoff_release(&power_mgr);
}


/**
 * Driver is asynchronous, so no waiting on the semaphore.
 */
bool nrf_axon_platform_reserve_for_driver(void)
{
	/*
	 * vote for power here even if the reservation failed. The pending power vote will
	 * prevent Axon from being powered down when the user releases Axon.
	 */
	axon_power_request();

	if (0 == k_sem_take(&axon_reserve_sem, K_NO_WAIT)) {
		/* driver will never reserve from its own thread, always from a user thread. */
		axon_owner_thread_id = AXON_DRIVER_THREAD_ID;
		return true;
	}
	return false;
}

/**
 * to use the hardware in synchronous mode, must reserve for exclusive use.
 * This function will be called by the driver during asynchronous inference when intrinsics are
 * invoked by the CPU. When this happens,the driver will already own the reservation, and the call
 * will always happen in the driver thread, so the driver appears as a user at this point and the
 * function will short-circuit.
 */
bool nrf_axon_platform_reserve_for_user(void)
{
	/* check if already own the reservation */
	if (k_current_get() == axon_owner_thread_id) {
		return true; /* already own it, short-circuit. */
	}
	/**
	 * if the driver thread ever requests to reserve for user it is because intrinsics
	 * are part of the command buffer and so axon is available
	 */
	if ((axon_owner_thread_id != NULL) && (k_current_get() == AXON_DRIVER_THREAD_ID)) {
		return true;
	}
	/*
	 * don't already own it, so need to wait.
	 * vote for power now to avoid a glitch when it is freed.
	 */
	axon_power_request();

	k_sem_take(&axon_reserve_sem, K_FOREVER);
	axon_owner_thread_id = k_current_get();
	return true;
}

/**
 * this only gets called from user threads. If a user is freeing the driver, the driver has priority
 * to get it.
 */
void nrf_axon_platform_free_reservation_from_user(void)
{
	axon_power_release();

	if (nrf_axon_queue_not_empty()) {
		/*
		 * driver needs the axon hardware, so don't free the sem, just start the
		 * hardware. It will have its own power vote pending.
		 */
		axon_owner_thread_id = AXON_DRIVER_THREAD_ID;
		nrf_axon_start_queue_processing();
		return;
	}
	axon_owner_thread_id = NULL;
	k_sem_give(&axon_reserve_sem);
}

void nrf_axon_platform_free_reservation_from_driver(void)
{
	axon_owner_thread_id = NULL;
	axon_power_release(); /* power down axon */
	k_sem_give(&axon_reserve_sem);
}

nrf_axon_result_e nrf_axon_platform_init(void)
{
	LOG_DBG("AXONS_BASE_ADDR: 0x%p", AXON_BASE_ADDR);
	nrf_axon_result_e result;

	static const struct onoff_transitions transitions = {
		.start = axon_power_on,
		.stop = axon_power_off,
	};
	int rc;

#if defined(CONFIG_SOC_NRF9251)
	/**
	 * @FIXME!! MAGIC SETTING TO ENABLE AXON'S RAM.
	 * SHOULD THIS BE SET ONCE OR TURNED ON/OFF EACH TIME AXON IS ENABLED?
	 */
	UICR_PERIPHCONF_ENTRY(PERIPHCONF_MEMCONF_POWER_CONTROL(0x5F8C7000, 0, 0x00c0ffff));

	rc = nrfs_mram_init(nrf_axon_mram_latency_evt_handler);
	if (rc != NRFS_SUCCESS) {
		return NRF_AXON_RESULT_FAILURE;
	}
#endif

	rc = onoff_manager_init(&power_mgr, &transitions);
	if (rc < 0) {
		return NRF_AXON_RESULT_FAILURE;
	}

	k_work_init(&axon_driver_work, axon_driver_work_handler);

	axon_enable();
	result = nrf_axon_driver_init(AXON_BASE_ADDR);
	if (result != NRF_AXON_RESULT_SUCCESS) {
		return result;
	}

	IRQ_CONNECT(AXON_IRQ_NO, AXON_IRQ_PRIORITY, axon_platform_irq_handler, 0, 0);
	enable_axon_interrupt();

	axon_disable();

	return NRF_AXON_RESULT_SUCCESS;
}


void nrf_axon_platform_close(void)
{
	axon_disable();
}
