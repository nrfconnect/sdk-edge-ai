/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "scenarios.h"

#include <zephyr/sys/util.h>

/*
 * Column order: walking, sitting, running, laying. Each row sums to 1.
 * The comment names the dominant (predicted) class of the row.
 */

const char *const activity_class_name[ACTIVITY_CLASS_COUNT] = {
	[ACTIVITY_CLASS_WALKING] = "walking",
	[ACTIVITY_CLASS_SITTING] = "sitting",
	[ACTIVITY_CLASS_RUNNING] = "running",
	[ACTIVITY_CLASS_LAYING] = "laying",
};

/* Scenario 1: walking 7 (35%), sitting 3 (15%), running 1 (5%), laying 9 (45%) */
static const float scenario_1[][ACTIVITY_CLASS_COUNT] = {
	{0.82f, 0.06f, 0.08f, 0.04f}, /* walking */
	{0.88f, 0.04f, 0.06f, 0.02f}, /* walking */
	{0.91f, 0.03f, 0.05f, 0.01f}, /* walking */
	{0.76f, 0.05f, 0.16f, 0.03f}, /* walking */
	{0.38f, 0.03f, 0.57f, 0.02f}, /* running */
	{0.71f, 0.05f, 0.21f, 0.03f}, /* walking */
	{0.86f, 0.06f, 0.06f, 0.02f}, /* walking */
	{0.64f, 0.27f, 0.04f, 0.05f}, /* walking */
	{0.18f, 0.69f, 0.02f, 0.11f}, /* sitting */
	{0.06f, 0.85f, 0.01f, 0.08f}, /* sitting */
	{0.03f, 0.72f, 0.01f, 0.24f}, /* sitting */
	{0.02f, 0.31f, 0.01f, 0.66f}, /* laying */
	{0.01f, 0.12f, 0.01f, 0.86f}, /* laying */
	{0.01f, 0.06f, 0.00f, 0.93f}, /* laying */
	{0.01f, 0.04f, 0.00f, 0.95f}, /* laying */
	{0.00f, 0.03f, 0.00f, 0.97f}, /* laying */
	{0.00f, 0.02f, 0.00f, 0.98f}, /* laying */
	{0.01f, 0.03f, 0.00f, 0.96f}, /* laying */
	{0.00f, 0.04f, 0.00f, 0.96f}, /* laying */
	{0.01f, 0.05f, 0.00f, 0.94f}, /* laying */
};

/* Scenario 2: walking 3 (15%), sitting 3 (15%), running 1 (5%), laying 13 (65%) */
static const float scenario_2[][ACTIVITY_CLASS_COUNT] = {
	{0.01f, 0.04f, 0.00f, 0.95f}, /* laying */
	{0.00f, 0.03f, 0.00f, 0.97f}, /* laying */
	{0.00f, 0.02f, 0.00f, 0.98f}, /* laying */
	{0.01f, 0.05f, 0.00f, 0.94f}, /* laying */
	{0.01f, 0.09f, 0.00f, 0.90f}, /* laying */
	{0.02f, 0.36f, 0.01f, 0.61f}, /* laying */
	{0.02f, 0.55f, 0.01f, 0.42f}, /* sitting */
	{0.01f, 0.30f, 0.00f, 0.69f}, /* laying */
	{0.01f, 0.11f, 0.00f, 0.88f}, /* laying */
	{0.00f, 0.05f, 0.00f, 0.95f}, /* laying */
	{0.00f, 0.03f, 0.00f, 0.97f}, /* laying */
	{0.00f, 0.02f, 0.00f, 0.98f}, /* laying */
	{0.01f, 0.04f, 0.00f, 0.95f}, /* laying */
	{0.01f, 0.21f, 0.00f, 0.78f}, /* laying */
	{0.04f, 0.68f, 0.01f, 0.27f}, /* sitting */
	{0.09f, 0.83f, 0.02f, 0.06f}, /* sitting */
	{0.58f, 0.35f, 0.04f, 0.03f}, /* walking */
	{0.84f, 0.07f, 0.07f, 0.02f}, /* walking */
	{0.79f, 0.03f, 0.17f, 0.01f}, /* walking */
	{0.29f, 0.02f, 0.68f, 0.01f}, /* running */
};

/* Scenario 3: walking 5 (25%), sitting 7 (35%), running 3 (15%), laying 5 (25%) */
static const float scenario_3[][ACTIVITY_CLASS_COUNT] = {
	{0.05f, 0.87f, 0.01f, 0.07f}, /* sitting */
	{0.04f, 0.91f, 0.01f, 0.04f}, /* sitting */
	{0.07f, 0.84f, 0.02f, 0.07f}, /* sitting */
	{0.22f, 0.71f, 0.03f, 0.04f}, /* sitting */
	{0.69f, 0.24f, 0.05f, 0.02f}, /* walking */
	{0.87f, 0.05f, 0.07f, 0.01f}, /* walking */
	{0.72f, 0.02f, 0.25f, 0.01f}, /* walking */
	{0.31f, 0.01f, 0.67f, 0.01f}, /* running */
	{0.12f, 0.01f, 0.86f, 0.01f}, /* running */
	{0.08f, 0.00f, 0.92f, 0.00f}, /* running */
	{0.55f, 0.02f, 0.42f, 0.01f}, /* walking */
	{0.81f, 0.09f, 0.08f, 0.02f}, /* walking */
	{0.26f, 0.66f, 0.02f, 0.06f}, /* sitting */
	{0.05f, 0.88f, 0.01f, 0.06f}, /* sitting */
	{0.03f, 0.62f, 0.00f, 0.35f}, /* sitting */
	{0.01f, 0.27f, 0.00f, 0.72f}, /* laying */
	{0.01f, 0.09f, 0.00f, 0.90f}, /* laying */
	{0.00f, 0.04f, 0.00f, 0.96f}, /* laying */
	{0.00f, 0.02f, 0.00f, 0.98f}, /* laying */
	{0.01f, 0.03f, 0.00f, 0.96f}, /* laying */
};

BUILD_ASSERT(ARRAY_SIZE(scenario_1) <= SCENARIO_MAX_VECTORS);
BUILD_ASSERT(ARRAY_SIZE(scenario_2) <= SCENARIO_MAX_VECTORS);
BUILD_ASSERT(ARRAY_SIZE(scenario_3) <= SCENARIO_MAX_VECTORS);

const struct scenario scenarios[SCENARIO_NUM] = {
	{.vectors = scenario_1, .num_vectors = ARRAY_SIZE(scenario_1)},
	{.vectors = scenario_2, .num_vectors = ARRAY_SIZE(scenario_2)},
	{.vectors = scenario_3, .num_vectors = ARRAY_SIZE(scenario_3)},
};
