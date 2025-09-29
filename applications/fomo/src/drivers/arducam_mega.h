
/**
 * Copyright (c) 2023 Arducam Technology Co., Ltd. <www.arducam.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_CAMERA_ARDUCAM_MEGA_H_
#define ZEPHYR_INCLUDE_DRIVERS_CAMERA_ARDUCAM_MEGA_H_

#ifdef __cplusplus
extern "C" {
#endif

enum {
	ARDUCAM_SENSOR_5MP_1 = 0x81,
	ARDUCAM_SENSOR_3MP_1 = 0x82,
	ARDUCAM_SENSOR_5MP_2 = 0x83, /* 2592x1936 */
	ARDUCAM_SENSOR_3MP_2 = 0x84,
};

/**
 * @enum MEGA_PIXELFORMAT
 * @brief Configure camera pixel format
 */
enum MEGA_PIXELFORMAT {
	MEGA_PIXELFORMAT_JPG = 0X01,
	MEGA_PIXELFORMAT_RGB565 = 0X02,
	MEGA_PIXELFORMAT_YUV = 0X03,
};

/**
 * @enum MEGA_RESOLUTION
 * @brief Configure camera resolution
 */
enum MEGA_RESOLUTION {
	MEGA_RESOLUTION_QQVGA = 0x00,   /**<160x120 */
	MEGA_RESOLUTION_QVGA = 0x01,    /**<320x240*/
	MEGA_RESOLUTION_VGA = 0x02,     /**<640x480*/
	MEGA_RESOLUTION_SVGA = 0x03,    /**<800x600*/
	MEGA_RESOLUTION_HD = 0x04,      /**<1280x720*/
	MEGA_RESOLUTION_SXGAM = 0x05,   /**<1280x960*/
	MEGA_RESOLUTION_UXGA = 0x06,    /**<1600x1200*/
	MEGA_RESOLUTION_FHD = 0x07,     /**<1920x1080*/
	MEGA_RESOLUTION_QXGA = 0x08,    /**<2048x1536*/
	MEGA_RESOLUTION_WQXGA2 = 0x09,  /**<2592x1944*/
	MEGA_RESOLUTION_96X96 = 0x0a,   /**<96x96*/
	MEGA_RESOLUTION_128X128 = 0x0b, /**<128x128*/
	MEGA_RESOLUTION_320X320 = 0x0c, /**<320x320*/
	MEGA_RESOLUTION_12 = 0x0d,      /**<Reserve*/
	MEGA_RESOLUTION_13 = 0x0e,      /**<Reserve*/
	MEGA_RESOLUTION_14 = 0x0f,      /**<Reserve*/
	MEGA_RESOLUTION_15 = 0x10,      /**<Reserve*/
	MEGA_RESOLUTION_NONE,
};

/**
 * @struct arducam_mega_info
 * @brief Some information about mega camera
 */
struct arducam_mega_info {
	int support_resolution;
	int support_special_effects;
	unsigned long exposure_value_max;
	unsigned int exposure_value_min;
	unsigned int gain_value_max;
	unsigned int gain_value_min;
	unsigned char enable_focus;
	unsigned char enable_sharpness;
	unsigned char device_address;
	unsigned char camera_id;
};

#ifdef __cplusplus
}
#endif

#endif /* ZEPHYR_INCLUDE_DRIVERS_CAMERA_ARDUCAM_MEGA_H_ */
