/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#ifndef AXON_LOW_POWER_INSTRUMENTATION_H
#define AXON_LOW_POWER_INSTRUMENTATION_H

/**
 * @brief Assert the per-inference trace line (infer-trace-gpios) active.
 *
 * Intended to bracket @c nrf_axon_nn_model_infer_sync(). Pin is driven high
 * (active per devicetree @c GPIO_ACTIVE_*).
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_GPIO_TRACING
 *       is disabled.
 */
void infer_trace_begin(void);

/**
 * @brief Release the per-inference trace line (infer-trace-gpios) to inactive.
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_GPIO_TRACING
 *       is disabled.
 */
void infer_trace_end(void);

/**
 * @brief Assert the sweep-envelope trace line (sweep-trace-gpios) active.
 *
 * Intended to wrap one full sliding-window sweep so the scope shows sweep
 * duration nested inside per-call infer_trace pulses.
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_GPIO_TRACING
 *       is disabled.
 */
void sweep_trace_begin(void);

/**
 * @brief Release the sweep trace line (sweep-trace-gpios) to inactive.
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_GPIO_TRACING
 *       is disabled.
 */
void sweep_trace_end(void);

/**
 * @brief Turn on the sweep-indication LED (devicetree led0) for the current sweep.
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_LED_INDICATION
 *       is disabled.
 */
void sweep_led_on(void);

/**
 * @brief Turn off the sweep-indication LED after a sweep.
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_LED_INDICATION
 *       is disabled.
 */
void sweep_led_off(void);

/**
 * @brief Start sweep-scope instrumentation (LED if enabled, then GPIO sweep trace if enabled).
 *
 * @note Safe to call with any Kconfig combination; no-op when all instrumentation is off.
 */
void inst_sweep_begin(void);

/**
 * @brief End sweep-scope instrumentation (GPIO sweep trace if enabled, then LED if enabled).
 *
 * Call on both success and error paths after a started sweep.
 */
void inst_sweep_end(void);

/**
 * @brief Start per-inference instrumentation (GPIO infer trace if enabled).
 */
void inst_infer_begin(void);

/**
 * @brief End per-inference instrumentation (GPIO infer trace if enabled).
 */
void inst_infer_end(void);

/**
 * @brief Configure optional sweep LED and trace GPIOs.
 *
 * Runs @c led.c then @c gpio_trace.c hardware setup when each option is enabled.
 *
 * @retval 0       Success.
 * @retval -ENODEV A GPIO device is not ready.
 * @retval other   Negative errno from @c gpio_pin_configure_dt() on failure.
 */
int inst_init(void);

/**
 * @brief Configure the GPIO tracing (infer-trace-gpios and sweep-trace-gpios).
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_GPIO_TRACING
 *       is disabled.
 */
int gpio_trace_hw_init(void);

/**
 * @brief Configure the sweep-indication LED (devicetree led0).
 *
 * @note Not available (implementation not built) if @c CONFIG_AXON_LOW_POWER_LED_INDICATION
 *       is disabled.
 */
int led_hw_init(void);

#endif /* AXON_LOW_POWER_INSTRUMENTATION_H */
