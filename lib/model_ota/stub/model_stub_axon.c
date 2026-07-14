/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 *
 * Compiled and linked completely separately from the application (see
 * cmake/nrf_axon_model_stub.cmake), at the model_storage partition's own address, with a
 * generated PROVIDE() linker script (see ../../../tools/model_ota/extract_elf_syms.py)
 * resolving every app-owned symbol the model header references to the *deployed* application's
 * real addresses for them.
 *
 * model_header_patched.h (found via -I<per-target work dir> on the compile command - see
 * ../cmake/nrf_axon_model_stub.cmake) is a patched copy of the actual generated model header
 * for whichever model this stub is building (see ../../../tools/model_ota/gen_axon_stub_fixups.py
 * for what "patched" means): which header that is is a per-model, per-sample choice, so it is a
 * build system concern, not something this file should hardcode - hence the fixed name every
 * per-target work dir uses instead. It is included last, after the platform headers it itself
 * depends on (int8_t, NULL, nrf_axon_nn_compiled_model_s, ...), rather than via -include on the
 * compile command, since -include inserts a header *before* anything else in the translation
 * unit - too early for any of those types/macros to exist yet.
 */
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_driver.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#ifndef MODEL_STUB_HEADER_RESERVE_LEN
#error "MODEL_STUB_HEADER_RESERVE_LEN must be defined (see nrf_axon_model_stub.cmake)"
#endif

/*
 * Reserves exactly the number of bytes package_model_axon.py's on-flash header (struct
 * model_pkg_axon_header) occupies, placed by model_stub_axon.ld right before the model header's
 * own data (see .model_stub_header_reserve there). This is what lets package_base - and
 * therefore every pointer field the model header's own data carries - already be correct for
 * where the flashed package's payload will actually sit (partition base + this header's size),
 * with package_model_axon.py only ever overwriting these placeholder bytes with the real header
 * it computes once every other byte is known (e.g. the CRC32), never shifting anything after
 * it.
 */
__attribute__((section(".model_stub_header_reserve"), used))
static const uint8_t model_stub_header_reserve[MODEL_STUB_HEADER_RESERVE_LEN];

#include "model_header_patched.h"
