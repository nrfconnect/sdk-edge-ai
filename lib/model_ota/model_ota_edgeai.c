/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Workarounds linked in with -Wl,--wrap=nrf_edgeai_init and -Wl,--wrap=nrf_edgeai_run_inference:
 * - nrf_edgeai_init() zeroes nrf_edgeai_t.state between nrf_edgeai_guard_lock() and
 *   nrf_edgeai_guard_unlock(), which drops MODEL_OTA_EDGEAI_STATE_OTA_MANAGED.
 * - Both return early on errors without unlocking, leaking the guard reader. In
 *   nrf_edgeai_run_inference() this includes the nested nrf_edgeai_process_features() lock
 *   failing once an update blocked the guard, which leaves the update unable to drain forever.
 * Remove once the runtime preserves user state bits and unlocks on every path.
 */

#include <model_ota/model_ota_edgeai.h>

#include <zephyr/sys/util.h>

#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
#include <model_ota/model_ota_guard.h>
#endif

nrf_edgeai_err_t __real_nrf_edgeai_init(nrf_edgeai_t *p_edgeai);
nrf_edgeai_err_t __wrap_nrf_edgeai_init(nrf_edgeai_t *p_edgeai);
nrf_edgeai_err_t __real_nrf_edgeai_run_inference(nrf_edgeai_t *p_edgeai);
nrf_edgeai_err_t __wrap_nrf_edgeai_run_inference(nrf_edgeai_t *p_edgeai);

static nrf_edgeai_err_t call_guarded(nrf_edgeai_t *p_edgeai,
				     nrf_edgeai_err_t (*fn)(nrf_edgeai_t *p_edgeai))
{
	nrf_edgeai_err_t err;
	uint32_t ota_bit;

	if (p_edgeai == NULL) {
		return fn(p_edgeai);
	}

	/* Hide the bit so the runtime's lock/unlock are no-ops; the reader is held here instead. */
	ota_bit = p_edgeai->state.rt.user_reserved & MODEL_OTA_EDGEAI_STATE_OTA_MANAGED;
	p_edgeai->state.rt.user_reserved &= ~MODEL_OTA_EDGEAI_STATE_OTA_MANAGED;

#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
	if (ota_bit != 0U && model_ota_guard_acquire() != 0) {
		p_edgeai->state.rt.user_reserved |= ota_bit;
		return NRF_EDGEAI_ERR_UNAVAILABLE;
	}
#endif

	err = fn(p_edgeai);
	p_edgeai->state.rt.user_reserved |= ota_bit;

#if IS_ENABLED(CONFIG_MODEL_OTA_SMP)
	if (ota_bit != 0U) {
		model_ota_guard_release();
	}
#endif

	return err;
}

nrf_edgeai_err_t __wrap_nrf_edgeai_init(nrf_edgeai_t *p_edgeai)
{
	return call_guarded(p_edgeai, __real_nrf_edgeai_init);
}

nrf_edgeai_err_t __wrap_nrf_edgeai_run_inference(nrf_edgeai_t *p_edgeai)
{
	return call_guarded(p_edgeai, __real_nrf_edgeai_run_inference);
}
