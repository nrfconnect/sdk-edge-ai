/*
 * Copyright (c) 2025 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/* This configuration file is included only once from runner and holds
 * information about prediction categories of machine learning model.
 */

/* This structure enforces the header file is included only once in the build.
 * Violating this requirement triggers a multiple definition error at link time.
 */
const struct {
} model_categories_def_include_once;

const char *model_categories[] = {"circle", "hold", "tap", "tilt", "updown"};
