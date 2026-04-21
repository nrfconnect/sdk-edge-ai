/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#pragma once

#ifdef CONFIG_AMBIENT_SENSING_MODEL_SNORING
#include <models/snoring/nrf_edgeai_user_model.h>
#elif CONFIG_AMBIENT_SENSING_MODEL_DOG_BARKING
#include <models/dog_barking/nrf_edgeai_user_model.h>
#elif CONFIG_AMBIENT_SENSING_MODEL_CAT_MEOWING
#include <models/cat_meowing/nrf_edgeai_user_model.h>
#elif CONFIG_AMBIENT_SENSING_MODEL_BABY_CRYING
#include <models/baby_crying/nrf_edgeai_user_model.h>
#endif

#ifdef CONFIG_AMBIENT_SENSING_MODEL_SNORING
static inline nrf_edgeai_t *get_ambient_sensing_model()
{
	return nrf_edgeai_user_model_snoring();
}
#elif CONFIG_AMBIENT_SENSING_MODEL_DOG_BARKING
static inline nrf_edgeai_t *get_ambient_sensing_model()
{
	return nrf_edgeai_user_model_dog_barking();
}
#elif CONFIG_AMBIENT_SENSING_MODEL_CAT_MEOWING
static inline nrf_edgeai_t *get_ambient_sensing_model()
{
	return nrf_edgeai_user_model_cat_meowing();
}
#elif CONFIG_AMBIENT_SENSING_MODEL_BABY_CRYING
static inline nrf_edgeai_t *get_ambient_sensing_model()
{
	return nrf_edgeai_user_model_baby_crying();
}
#endif
