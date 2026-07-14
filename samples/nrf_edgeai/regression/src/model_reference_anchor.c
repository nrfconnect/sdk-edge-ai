/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/*
 * Reference-build-only translation unit (see CONFIG_NRF_EDGEAI_REGRESSION_REFERENCE_BUILD in
 * Kconfig): never compiled into the deployed app, which no longer links in the generated Axon
 * model sources - it loads its model from the model_storage flash partition at runtime instead.
 * Packaging a model update still needs a real link address for the model's constants blob (see
 * package_base in model_pkg_axon_header), which only exists in a build that actually references
 * those sources.
 *
 * Including the header here is enough to make the compiler emit model_axon_user_instance_36025,
 * cmd_buffer_axon_user_instance_36025 and axon_model_const_axon_user_instance_36025;
 * CMakeLists.txt's toolchain_ld_force_undefined_symbols(model_axon_user_instance_36025) then
 * keeps that symbol's section (and, transitively via its own relocations to the other two,
 * theirs as well) alive through the linker's --gc-sections, since nothing else in this
 * reference-only image references any of them. This build does nothing else and is never
 * flashed as the application - see README.rst.
 */
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <axon/nrf_axon_platform.h>
#include <drivers/axon/nrf_axon_nn_infer.h>

#include "nrf_edgeai_user_model_axon.h"
