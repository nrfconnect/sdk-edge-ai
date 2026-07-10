/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <stddef.h>

#include <axon/nrf_axon_platform.h>
#include <axon/nrf_axon_model_partition.h>

bool nrf_axon_model_partition_is_valid(uintptr_t base_addr)
{
	const struct nrf_axon_model_partition_header *header =
		(const struct nrf_axon_model_partition_header *)base_addr;

	return header->magic == NRF_AXON_MODEL_PARTITION_MAGIC &&
	       header->version == NRF_AXON_MODEL_PARTITION_VERSION &&
	       header->model_offset >= sizeof(*header);
}

const nrf_axon_nn_compiled_model_s *nrf_axon_model_partition_get(uintptr_t base_addr)
{
	const struct nrf_axon_model_partition_header *header;

#if (NRF_AXON_INTERLAYER_BUFFER_SIZE > 0)
	/* Keep the interlayer buffer linked for partition models that reference it by address. */
	__asm__ volatile("" : : "r"(&nrf_axon_interlayer_buffer[0]) : "memory");
#endif

	if (!nrf_axon_model_partition_is_valid(base_addr)) {
		return NULL;
	}

	header = (const struct nrf_axon_model_partition_header *)base_addr;

	return (const nrf_axon_nn_compiled_model_s *)(base_addr + header->model_offset);
}
