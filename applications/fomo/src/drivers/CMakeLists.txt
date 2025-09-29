# SPDX-License-Identifier: Apache-2.0

zephyr_library_named(local_drivers)

zephyr_library_sources_ifdef(CONFIG_VIDEO_ARDUCAM_MEGA arducam_mega.c)
