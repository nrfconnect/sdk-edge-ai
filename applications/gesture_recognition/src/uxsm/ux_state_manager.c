/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "ux_state_manager.h"

#include <errno.h>
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/util.h>

#include "button/button.h"
#include "hid/ble_hid.h"

LOG_MODULE_REGISTER(uxsm, CONFIG_LOG_DEFAULT_LEVEL);

/**
 * @brief Mutable UX state, grouped for clarity and accessed via atomic operations.
 *
 * - @c btn_func is written from multiple contexts (BT stack callbacks running on
 *   the system workqueue or BT RX thread, via @ref uxsm_set_btn_func)
 *   and read concurrently from the button click handler. Atomic access guards
 *   against writer/reader and writer/writer races.
 *
 * - @c remotectrl_mode is written only from the button click handler (single
 *   context), but read from other threads (e.g. the LED update thread). Atomic
 *   access here is purely for reader visibility and memory ordering.
 */
typedef struct {
	atomic_t btn_func;	  /* ux_btn_func_t: determines what button presses do */
	atomic_t remotectrl_mode; /* ux_remotectrl_mode_t: active remote-control profile */
} ux_app_state_t;

static ux_app_state_t ux_app_state = {
	.btn_func = ATOMIC_INIT(UX_BTN_FUNC_NORMAL),
	.remotectrl_mode = ATOMIC_INIT(UX_REMOTECTRL_MODE_MUSIC),
};

/* ---------------------------------------------------------------------------
 * Action table: single source of truth for (btn_func, click) -> behavior.
 *
 * Each cell pairs the handler that runs when a click happens with the help
 * text printed in the instructions banner. The two cannot drift out of sync
 * because they live together.
 *
 * To add a new state: add an enum value, define its handlers, and add a row.
 * To add a new click type: extend click_entry_t and btn_func_entry_t, and add
 * a column in every row.
 * -------------------------------------------------------------------------
 */

typedef void (*click_action_fn_t)(void);

typedef struct {
	click_action_fn_t action; /* NULL = no-op on this click */
	const char *help;	  /* NULL = omit from help banner */
	const char *log_msg;	  /* NULL = don't log on click */
} click_entry_t;

typedef struct {
	const char *name;   /* e.g. "NORMAL"; NULL marks the slot unused */
	const char *banner; /* Banner title in help output */
	click_entry_t on_short;
	click_entry_t on_long;
} btn_func_entry_t;

/* --- Action handlers ----------------------------------------------------- */

static void act_toggle_remotectrl_mode(void)
{
	ux_remotectrl_mode_t current = atomic_get(&ux_app_state.remotectrl_mode);
	ux_remotectrl_mode_t new_mode = (current == UX_REMOTECTRL_MODE_MUSIC)
						? UX_REMOTECTRL_MODE_PRESENTATION
						: UX_REMOTECTRL_MODE_MUSIC;

	ux_remotectrl_mode_t old = (ux_remotectrl_mode_t)atomic_set(&ux_app_state.remotectrl_mode,
								    (atomic_val_t)new_mode);

	if (old == new_mode) {
		return;
	}

	/* In HID builds the mode drives the keymap used by send_bt_keyboard_key(),
	 * so the user-facing name ("MUSIC"/"PRESENTATION") is meaningful.
	 */
	LOG_INF("Remote-control mode -> %s",
		(new_mode == UX_REMOTECTRL_MODE_MUSIC) ? "MUSIC" : "PRESENTATION");
}

static void act_forget_bonds(void)
{
	int err = ble_hid_forget_bonds();

	if (err != 0) {
		LOG_ERR("Failed to forget bonds (err %d)", err);
	}
}

static void act_confirm_pairing(void)
{
	int err = ble_hid_confirm_pairing(true);

	if (err != 0) {
		LOG_ERR("Failed to confirm pairing (err %d)", err);
	}
}

static void act_reject_pairing(void)
{
	int err = ble_hid_confirm_pairing(false);

	if (err != 0) {
		LOG_ERR("Failed to reject pairing (err %d)", err);
	}
}

/* --- The table ----------------------------------------------------------- */

static const btn_func_entry_t BTN_FUNC_TABLE[] = {
	[UX_BTN_FUNC_NORMAL] = {
			.name = "NORMAL",
			.banner = "Button Functionality (Normal Mode)",
			.on_short = {
					.action = act_toggle_remotectrl_mode,
					.help = "Toggle remote-control mode (Presentation <-> "
						"Music)",
					.log_msg = "Normal mode: SHORT click -> toggle "
						   "remote-control mode",
				},
			.on_long = {
					.action = act_forget_bonds,
					.help = "Forget bonded devices and disconnect",
					.log_msg =
						"Normal mode: LONG click -> forget bonded devices",
				},
		},
	[UX_BTN_FUNC_PAIRING] = {
			.name = "PAIRING",
			.banner = "Button Functionality (Pairing Confirmation Mode)",
			.on_short = {
					.action = act_reject_pairing,
					.help = "Reject pairing",
					.log_msg = "Pairing mode: SHORT click -> reject pairing",
				},
			.on_long = {
					.action = act_confirm_pairing,
					.help = "Confirm pairing",
					.log_msg = "Pairing mode: LONG click -> confirm pairing",
				},
		},
};

/* --- Table helpers ------------------------------------------------------- */

static const btn_func_entry_t *btn_func_entry(ux_btn_func_t fn)
{
	if ((size_t)fn >= ARRAY_SIZE(BTN_FUNC_TABLE)) {
		return NULL;
	}
	const btn_func_entry_t *e = &BTN_FUNC_TABLE[fn];

	/* .name == NULL indicates an unused slot (e.g. PAIRING when HID is off). */
	return (e->name != NULL) ? e : NULL;
}

static const click_entry_t *click_entry_for(const btn_func_entry_t *e, button_click_t click)
{
	switch (click) {
	case BUTTON_CLICK_SHORT:
		return &e->on_short;
	case BUTTON_CLICK_LONG:
		return &e->on_long;
	default:
		return NULL;
	}
}

static void print_instructions(const btn_func_entry_t *e)
{
	__ASSERT_NO_MSG(e != NULL);

	LOG_INF("===== %s =====", e->banner);
	LOG_INF("Short press (< %u ms): %s", BUTTON_SHORT_CLICK_MSEC,
		e->on_short.help ? e->on_short.help : "no action");
	LOG_INF("Long press  (> %u ms): %s", BUTTON_LONG_CLICK_MSEC,
		e->on_long.help ? e->on_long.help : "no action");
	LOG_INF("==============================================");
}

/* --- Button dispatch ----------------------------------------------------- */

static void on_button_click(button_click_t click)
{
	ux_btn_func_t fn = (ux_btn_func_t)atomic_get(&ux_app_state.btn_func);
	const btn_func_entry_t *e = btn_func_entry(fn);

	if (e == NULL) {
		LOG_ERR("Unhandled btn_func %d", (int)fn);
		return;
	}

	const click_entry_t *c = click_entry_for(e, click);

	if (c == NULL) {
		return; /* click type not handled in this state */
	}

	if (c->log_msg != NULL) {
		LOG_INF("%s", c->log_msg);
	}

	if (c->action != NULL) {
		c->action();
	}
}

/* --- Public API ---------------------------------------------------------- */

int uxsm_init(void)
{
	/* Seems redundant given ATOMIC_INIT above, but kept defensively so that
	 * re-invocation (e.g. after a soft reset path that does not re-run
	 * static initializers) leaves the module in a known state.
	 */
	atomic_set(&ux_app_state.btn_func, (atomic_val_t)UX_BTN_FUNC_NORMAL);
	atomic_set(&ux_app_state.remotectrl_mode, (atomic_val_t)UX_REMOTECTRL_MODE_MUSIC);

	button_reg_click_handler(on_button_click);

	const btn_func_entry_t *e = btn_func_entry(UX_BTN_FUNC_NORMAL);

	__ASSERT_NO_MSG(e != NULL);
	print_instructions(e);

	return 0;
}

void uxsm_set_btn_func(ux_btn_func_t btn_func)
{
	const btn_func_entry_t *e = btn_func_entry(btn_func);

	if (e == NULL) {
		LOG_ERR("Attempt to set unknown btn_func %d", (int)btn_func);
		return;
	}

	ux_btn_func_t old =
		(ux_btn_func_t)atomic_set(&ux_app_state.btn_func, (atomic_val_t)btn_func);

	if (old == btn_func) {
		return;
	}

	LOG_INF("State transition -> %s", e->name);
	print_instructions(e);
}

ux_btn_func_t uxsm_get_btn_func(void)
{
	return (ux_btn_func_t)atomic_get(&ux_app_state.btn_func);
}

ux_remotectrl_mode_t uxsm_get_remotectrl_mode(void)
{
	return (ux_remotectrl_mode_t)atomic_get(&ux_app_state.remotectrl_mode);
}
