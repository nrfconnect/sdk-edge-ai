/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include <model_ota/model_pkg.h>

#include <string.h>

#include <axon/nrf_axon_platform.h>

#include <zephyr/logging/log.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/crc.h>
#include <zephyr/toolchain.h>

LOG_MODULE_REGISTER(model_pkg_axon, CONFIG_MODEL_OTA_LOG_LEVEL);

/*
 * nrf_axon_nn_compiled_model_s::model_name is a bare `const char *`. The packaged struct never
 * carries a usable value for it (resolving where an arbitrary string literal ends up in the
 * model stub's .rodata, reliably, is not worth the complexity for a PoC), so it is always
 * repointed here at the package's own name field instead.
 */
static char model_name_buf[MODEL_PKG_NAME_LEN + 1];

static bool magic_is_valid(const struct model_pkg_axon_header *hdr)
{
	return hdr->magic[0] == MODEL_PKG_MAGIC0 && hdr->magic[1] == MODEL_PKG_MAGIC1 &&
	       hdr->magic[2] == MODEL_PKG_MAGIC2 && hdr->magic[3] == MODEL_PKG_MAGIC3;
}

int model_pkg_load_axon(uint8_t fa_id, const uint8_t *partition_addr,
			 nrf_axon_nn_compiled_model_s *out_model,
			 struct model_pkg_axon_info *out_info)
{
	const struct flash_area *fa;
	struct model_pkg_axon_header hdr;
	int rc;

	rc = flash_area_open(fa_id, &fa);
	if (rc != 0) {
		LOG_ERR("Cannot open model_storage partition (err %d)", rc);
		return MODEL_PKG_ERR_NO_PARTITION;
	}

	rc = flash_area_read(fa, 0, &hdr, sizeof(hdr));
	if (rc != 0) {
		flash_area_close(fa);
		LOG_ERR("Flash read of package header failed (err %d)", rc);
		return MODEL_PKG_ERR_FLASH_READ;
	}

	if (!magic_is_valid(&hdr)) {
		flash_area_close(fa);
		LOG_WRN("No valid model package in model_storage (bad magic)");
		return MODEL_PKG_ERR_BAD_MAGIC;
	}

	if (hdr.format_version != MODEL_PKG_FORMAT_VERSION) {
		flash_area_close(fa);
		LOG_ERR("Unsupported package format version %u", hdr.format_version);
		return MODEL_PKG_ERR_BAD_FORMAT_VERSION;
	}

	if (hdr.model_type != MODEL_PKG_TYPE_AXON) {
		flash_area_close(fa);
		LOG_ERR("Package is not an Axon model (type %u)", hdr.model_type);
		return MODEL_PKG_ERR_WRONG_TYPE;
	}

	if (hdr.struct_size != sizeof(nrf_axon_nn_compiled_model_s)) {
		flash_area_close(fa);
		LOG_ERR("Package's nrf_axon_nn_compiled_model_s is %u B, this firmware's is %u B "
			"- packaging tool and firmware were built against different SDK "
			"versions",
			hdr.struct_size, (unsigned)sizeof(nrf_axon_nn_compiled_model_s));
		return MODEL_PKG_ERR_BAD_STRUCT_SIZE;
	}

	if (hdr.struct_offset > hdr.payload_size ||
	    hdr.struct_size > hdr.payload_size - hdr.struct_offset) {
		flash_area_close(fa);
		LOG_ERR("struct_offset/struct_size (%u/%u) fall outside payload_size (%u)",
			hdr.struct_offset, hdr.struct_size, hdr.payload_size);
		return MODEL_PKG_ERR_BAD_SECTION_LEN;
	}
	if (hdr.payload_size > fa->fa_size - sizeof(hdr)) {
		flash_area_close(fa);
		LOG_ERR("Package payload (%u B) exceeds model_storage capacity (%u B)",
			hdr.payload_size, (unsigned)(fa->fa_size - sizeof(hdr)));
		return MODEL_PKG_ERR_TOO_LARGE;
	}

	/* Everything from here on references the partition via its memory-mapped address
	 * (partition_addr), not the flash_area API, so the area doesn't need to stay open.
	 */
	flash_area_close(fa);

	const uint8_t *payload_ptr = partition_addr + sizeof(hdr);
	const uint8_t *struct_ptr = payload_ptr + hdr.struct_offset;

	/*
	 * package_base is where the packaging tool's model stub was linked (i.e. right where
	 * the payload actually starts once this header is flashed ahead of it), and every
	 * pointer field baked into the payload - flash-owned or app-owned - was resolved
	 * against that address. If it doesn't match where the payload actually lands in this
	 * partition, the package was built for a different partition layout (wrong --address,
	 * or a different board) and those pointers would be silently wrong - refuse to wire it
	 * up rather than risk that.
	 */
	uint32_t actual_package_base = (uint32_t)(uintptr_t)payload_ptr;

	if (hdr.package_base != actual_package_base) {
		LOG_ERR("Package built for payload at 0x%08x, but this partition places it at "
			"0x%08x - re-run package_model_axon.py with the matching --address",
			hdr.package_base, actual_package_base);
		return MODEL_PKG_ERR_BAD_PACKAGE_BASE;
	}

	/*
	 * Validate CRC32 by streaming straight from the memory-mapped partition: header (with
	 * crc32 zeroed) + the whole payload blob, in one pass - there is no per-section
	 * breakdown to track separately any more, since the payload is the model stub's own
	 * linked memory image, copied verbatim.
	 */
	uint32_t stored_crc = hdr.crc32;
	struct model_pkg_axon_header hdr_for_crc = hdr;

	hdr_for_crc.crc32 = 0;
	uint32_t computed_crc = crc32_ieee_update(0, (const uint8_t *)&hdr_for_crc, sizeof(hdr_for_crc));

	computed_crc = crc32_ieee_update(computed_crc, payload_ptr, hdr.payload_size);

	if (computed_crc != stored_crc) {
		LOG_ERR("Package CRC mismatch (stored 0x%08x, computed 0x%08x)", stored_crc,
			computed_crc);
		return MODEL_PKG_ERR_BAD_CRC;
	}

	nrf_axon_nn_compiled_model_s built;

	memcpy(&built, struct_ptr, sizeof(built));

	/* Sanity-check the struct's own flash-owned pointers land within this package's
	 * payload - a cheap defense-in-depth check against a corrupted or mismatched package,
	 * now that package_base (checked above) guarantees what address range the payload
	 * occupies.
	 */
	if ((const uint8_t *)built.cmd_buffer_ptr < payload_ptr ||
	    (const uint8_t *)built.cmd_buffer_ptr >= payload_ptr + hdr.payload_size ||
	    (const uint8_t *)built.model_const_ptr < payload_ptr ||
	    (const uint8_t *)built.model_const_ptr >= payload_ptr + hdr.payload_size) {
		LOG_ERR("Packaged struct's cmd_buffer_ptr/model_const_ptr do not fall within "
			"this package's payload - package is corrupted");
		return MODEL_PKG_ERR_BAD_PACKAGE_BASE;
	}

	/* Every RAM-owned pointer field (inputs[i].ptr for external inputs, output_ptr,
	 * persistent_vars, and op-extension function pointers embedded in cmd_buffer) already
	 * targets this application's real address: the model stub that produced this package
	 * was linked against this exact application's own zephyr.elf (see
	 * lib/model_ota/cmake/nrf_axon_model_stub.cmake), so there is nothing left to patch here.
	 *
	 * Dedicated packed-output buffers are the one field the model stub mechanism does not
	 * cover (see package_model_axon.py's validate_shape()): they are compile-time arrays
	 * declared by the generated model header itself, which the deployed app never links in.
	 */
	built.packed_output_buf = NULL;

	memset(model_name_buf, 0, sizeof(model_name_buf));
	memcpy(model_name_buf, hdr.name, MODEL_PKG_NAME_LEN);
	built.model_name = model_name_buf;

	nrf_axon_result_e vres = nrf_axon_nn_model_validate(&built);

	if (vres != NRF_AXON_RESULT_SUCCESS) {
		LOG_ERR("Loaded Axon model failed nrf_axon_nn_model_validate() (err %d)", vres);
		return MODEL_PKG_ERR_AXON_VALIDATE_FAILED;
	}

	*out_model = built;

	if (out_info != NULL) {
		memset(out_info->name, 0, sizeof(out_info->name));
		memcpy(out_info->name, hdr.name, MODEL_PKG_NAME_LEN);
		out_info->version = hdr.model_version;
		out_info->cmd_buffer_len = built.cmd_buffer_len;
		out_info->model_const_size = built.model_const_size;
	}

	LOG_INF("Loaded Axon model '%.*s' v0x%08x (%u cmd words, %u B const)", MODEL_PKG_NAME_LEN,
		hdr.name, hdr.model_version, built.cmd_buffer_len, built.model_const_size);

	return MODEL_PKG_OK;
}
