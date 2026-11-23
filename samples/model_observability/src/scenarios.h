/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

/**
 * @file scenarios.h
 * @defgroup sample_mo_scenarios Activity recognition scenarios
 * @{
 * @ingroup sample_mo
 *
 * Predefined class-probability vectors that stand in for the output of an
 * activity recognition model. Each scenario is fed to the observability
 * metrics as one Memfault CDR.
 */

#ifndef __SCENARIOS_H__
#define __SCENARIOS_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Activity classes, in probability vector order */
enum activity_class {
	ACTIVITY_CLASS_WALKING,
	ACTIVITY_CLASS_SITTING,
	ACTIVITY_CLASS_RUNNING,
	ACTIVITY_CLASS_LAYING,
	ACTIVITY_CLASS_COUNT,
};

/** Largest number of probability vectors in one scenario */
#define SCENARIO_MAX_VECTORS (20U)

/** Number of predefined scenarios */
#define SCENARIO_NUM (3U)

/** One scenario: a sequence of class-probability vectors */
struct scenario {
	/** Probability per class, one row per inference */
	const float (*vectors)[ACTIVITY_CLASS_COUNT];
	/** Number of rows in @ref vectors */
	size_t num_vectors;
};

/** Class names, indexed by @ref activity_class */
extern const char *const activity_class_name[ACTIVITY_CLASS_COUNT];

/** Predefined scenarios, fed in this order */
extern const struct scenario scenarios[SCENARIO_NUM];

#ifdef __cplusplus
}
#endif

#endif /* __SCENARIOS_H__ */

/**
 * @}
 */
