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

/* Same dedicated "model_storage" partition used by the Neuton loader; a device only ever runs
 * one backend's loader, so there is no risk of the two colliding on it.
 */
#define MODEL_PARTITION_ID PARTITION_ID(model_partition)

/*
 * All flash-owned sections (the struct itself, cmd_buffer, model_const, extra_outputs) are
 * referenced directly in this memory-mapped partition (no RAM copy of any of them, other than
 * the fixed-size struct copy into *out_model itself): the packaging tool already rewrote every
 * pointer field that targets one of them to its final flash address, so there is nothing left
 * to relocate at load time beyond the handful of app-RAM pointers handled below. Only valid
 * because model_partition is a "zephyr,mapped-partition" node (directly CPU-addressable), which
 * PARTITION_ADDRESS() relies on.
 */
#define MODEL_PARTITION_ADDR ((const uint8_t *)PARTITION_ADDRESS(model_partition))

/*
 * nrf_axon_nn_compiled_model_s::model_name is a bare `const char *`. The packaged struct never
 * carries a usable value for it (see package_model_axon.py: resolving where an arbitrary string
 * literal ends up in a reference build's .rodata, reliably, is not worth the complexity for a
 * PoC), so it is always repointed here at the package's own name field instead.
 */
static char model_name_buf[MODEL_PKG_NAME_LEN + 1];

static bool magic_is_valid(const struct model_pkg_axon_header *hdr)
{
	return hdr->magic[0] == MODEL_PKG_MAGIC0 && hdr->magic[1] == MODEL_PKG_MAGIC1 &&
	       hdr->magic[2] == MODEL_PKG_MAGIC2 && hdr->magic[3] == MODEL_PKG_MAGIC3;
}

/*
 * Adds an offset (smuggled through a pointer-sized struct field while the model sits in flash -
 * see package_model_axon.py's classification of "app RAM" pointer fields) to a RAM base address
 * this firmware actually has. The packaging tool cannot know that address itself: it may differ
 * between the reference build it ran against and whatever build is actually deployed.
 */
static inline int8_t *rebase_to_ram(void *field_as_offset, void *ram_base)
{
	uintptr_t offset = (uintptr_t)field_as_offset;

	return (int8_t *)ram_base + offset;
}

int model_pkg_load_axon(nrf_axon_nn_compiled_model_s *out_model,
			 struct model_pkg_axon_info *out_info)
{
	const struct flash_area *fa;
	struct model_pkg_axon_header hdr;
	int rc;

	rc = flash_area_open(MODEL_PARTITION_ID, &fa);
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

	if (hdr.section_len[MODEL_PKG_AXON_SEC_MODEL_STRUCT] != hdr.struct_size) {
		flash_area_close(fa);
		LOG_ERR("MODEL_STRUCT section length (%u) does not match struct_size (%u)",
			hdr.section_len[MODEL_PKG_AXON_SEC_MODEL_STRUCT], hdr.struct_size);
		return MODEL_PKG_ERR_BAD_SECTION_LEN;
	}

	uint32_t section_total = 0;

	for (size_t i = 0; i < MODEL_PKG_AXON_SECTION_COUNT; i++) {
		section_total += hdr.section_len[i];
	}
	if (section_total != hdr.payload_size) {
		flash_area_close(fa);
		LOG_ERR("Section lengths (%u) do not add up to payload_size (%u)", section_total,
			hdr.payload_size);
		return MODEL_PKG_ERR_BAD_SECTION_LEN;
	}

	uint32_t cmd_buffer_bytes = hdr.section_len[MODEL_PKG_AXON_SEC_CMD_BUFFER];
	uint32_t model_const_size = hdr.section_len[MODEL_PKG_AXON_SEC_MODEL_CONST];
	uint32_t extra_outputs_bytes = hdr.section_len[MODEL_PKG_AXON_SEC_EXTRA_OUTPUTS];
	uint32_t labels_bytes = hdr.section_len[MODEL_PKG_AXON_SEC_LABELS];
	uint32_t label_strings_bytes = hdr.section_len[MODEL_PKG_AXON_SEC_LABEL_STRINGS];

	if (cmd_buffer_bytes % sizeof(uint32_t) != 0) {
		flash_area_close(fa);
		LOG_ERR("cmd_buffer section length (%u) is not a multiple of %u bytes",
			cmd_buffer_bytes, (unsigned)sizeof(uint32_t));
		return MODEL_PKG_ERR_BAD_SECTION_LEN;
	}
	if (hdr.payload_size > fa->fa_size - sizeof(hdr)) {
		flash_area_close(fa);
		LOG_ERR("Package payload (%u B) exceeds model_storage capacity (%u B)",
			hdr.payload_size, (unsigned)(fa->fa_size - sizeof(hdr)));
		return MODEL_PKG_ERR_TOO_LARGE;
	}

	/* Everything from here on references the partition via its memory-mapped address
	 * (MODEL_PARTITION_ADDR), not the flash_area API, so the area doesn't need to stay open.
	 */
	flash_area_close(fa);

	uint32_t struct_offset = sizeof(hdr);
	uint32_t cmd_buffer_offset = struct_offset + hdr.section_len[MODEL_PKG_AXON_SEC_MODEL_STRUCT];
	uint32_t model_const_offset = cmd_buffer_offset + cmd_buffer_bytes;
	uint32_t extra_outputs_offset = model_const_offset + model_const_size;
	uint32_t labels_offset = extra_outputs_offset + extra_outputs_bytes;
	uint32_t label_strings_offset = labels_offset + labels_bytes;

	const uint8_t *struct_ptr = MODEL_PARTITION_ADDR + struct_offset;
	const uint32_t *cmd_buffer_ptr =
		(const uint32_t *)(MODEL_PARTITION_ADDR + cmd_buffer_offset);
	const void *model_const_ptr = MODEL_PARTITION_ADDR + model_const_offset;
	const void *extra_outputs_ptr = MODEL_PARTITION_ADDR + extra_outputs_offset;
	const void *labels_ptr = MODEL_PARTITION_ADDR + labels_offset;
	const void *label_strings_ptr = MODEL_PARTITION_ADDR + label_strings_offset;

	/*
	 * package_base is where the packaging tool assumed the MODEL_STRUCT section would land,
	 * and baked every flash-owned pointer field (cmd_buffer_ptr, model_const_ptr,
	 * extra_outputs, and cmd_buffer's own internal pointers into model_const) accordingly.
	 * If it doesn't match where this section actually is in this partition, the package was
	 * built for a different model_storage layout (wrong --address, or a different board)
	 * and those pointers would be silently wrong - refuse to wire it up rather than risk
	 * that.
	 */
	uint32_t actual_package_base = (uint32_t)(uintptr_t)struct_ptr;

	if (hdr.package_base != actual_package_base) {
		LOG_ERR("Package built for MODEL_STRUCT at 0x%08x, but this partition places it "
			"at 0x%08x - re-run package_model_axon.py with the matching --address",
			hdr.package_base, actual_package_base);
		return MODEL_PKG_ERR_BAD_PACKAGE_BASE;
	}

	/*
	 * cmd_buffer embeds literal references to nrf_axon_interlayer_buffer (for inter-layer
	 * data handoff) baked in by the reference build that produced this package - unlike
	 * model_const, this loader does not relocate them (see the module comment above). If
	 * this device's actual buffer isn't at the address the packaging tool saw, those
	 * embedded references are silently wrong: the NPU would still run, but would read and
	 * write whatever happens to live at the reference build's address instead of the real
	 * buffer, producing plausible-looking but incorrect predictions rather than an obvious
	 * failure. Refuse to wire up the model rather than risk that.
	 */
	uint32_t actual_interlayer_addr = (uint32_t)(uintptr_t)nrf_axon_interlayer_buffer;

	if (hdr.interlayer_addr != actual_interlayer_addr) {
		LOG_ERR("Package built for nrf_axon_interlayer_buffer at 0x%08x, but this "
			"device's is at 0x%08x - cmd_buffer's embedded references to it would "
			"be wrong; rebuild and repackage from a reference build that matches "
			"this firmware's memory layout",
			hdr.interlayer_addr, actual_interlayer_addr);
		return MODEL_PKG_ERR_INTERLAYER_MISMATCH;
	}

	/*
	 * Validate CRC32 by streaming straight from the memory-mapped partition: header (with
	 * crc32 zeroed) + struct + cmd_buffer + model_const + extra_outputs + labels +
	 * label_strings (each of the last three only if present).
	 */
	uint32_t stored_crc = hdr.crc32;
	struct model_pkg_axon_header hdr_for_crc = hdr;

	hdr_for_crc.crc32 = 0;
	uint32_t computed_crc = crc32_ieee_update(0, (const uint8_t *)&hdr_for_crc, sizeof(hdr_for_crc));

	computed_crc = crc32_ieee_update(computed_crc, struct_ptr, hdr.struct_size);
	computed_crc = crc32_ieee_update(computed_crc, (const uint8_t *)cmd_buffer_ptr,
					  cmd_buffer_bytes);
	computed_crc = crc32_ieee_update(computed_crc, model_const_ptr, model_const_size);
	if (extra_outputs_bytes != 0) {
		computed_crc = crc32_ieee_update(computed_crc, extra_outputs_ptr,
						  extra_outputs_bytes);
	}
	if (labels_bytes != 0) {
		computed_crc = crc32_ieee_update(computed_crc, labels_ptr, labels_bytes);
	}
	if (label_strings_bytes != 0) {
		computed_crc = crc32_ieee_update(computed_crc, label_strings_ptr,
						  label_strings_bytes);
	}

	if (computed_crc != stored_crc) {
		LOG_ERR("Package CRC mismatch (stored 0x%08x, computed 0x%08x)", stored_crc,
			computed_crc);
		return MODEL_PKG_ERR_BAD_CRC;
	}

	nrf_axon_nn_compiled_model_s built;

	memcpy(&built, struct_ptr, sizeof(built));

	/* Defense-in-depth: package_model_axon.py already refuses to package models shaped like
	 * this (see its module docstring), but a hand-crafted or corrupted package could still
	 * claim struct_size matches while carrying a shape this loader never patches correctly.
	 */
	if (built.persistent_vars.count != 0) {
		LOG_ERR("Package uses persistent_vars, which this loader does not support "
			"(persistent_vars.count=%u)", built.persistent_vars.count);
		return MODEL_PKG_ERR_UNSUPPORTED_SHAPE;
	}

	/* Sanity-check the flash-owned pointers the struct itself carries: they must already
	 * point exactly where this loader independently computed those sections to be, since
	 * package_base (checked above) is what the packaging tool based them on.
	 */
	if (built.cmd_buffer_ptr != cmd_buffer_ptr || built.model_const_ptr != model_const_ptr) {
		LOG_ERR("Packaged struct's cmd_buffer_ptr/model_const_ptr do not match this "
			"package's own section layout - package is corrupted");
		return MODEL_PKG_ERR_BAD_PACKAGE_BASE;
	}
	if (built.labels != NULL && (labels_bytes == 0 || (const void *)built.labels != labels_ptr)) {
		LOG_ERR("Packaged struct's labels pointer does not match this package's own "
			"LABELS section - package is corrupted");
		return MODEL_PKG_ERR_BAD_PACKAGE_BASE;
	}

	/* Patch the small, fixed set of pointer fields that refer to *this device's* RAM rather
	 * than flash-owned model data. Each was serialized as a byte offset (not an address) by
	 * package_model_axon.py, since the reference build's own RAM addresses are not
	 * necessarily the deployed app's.
	 */
	for (uint8_t i = 0; i < built.input_cnt && i < NRF_AXON_NN_MAX_MODEL_INPUTS; i++) {
		if (built.inputs[i].is_external) {
			built.inputs[i].ptr =
				rebase_to_ram(built.inputs[i].ptr, nrf_axon_interlayer_buffer);
		}
	}
	built.output_ptr = rebase_to_ram(built.output_ptr, nrf_axon_interlayer_buffer);
	/* Dedicated packed-output buffers are compile-time arrays declared by the generated
	 * model header, which the deployed app never links in - unsupported by this PoC.
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
