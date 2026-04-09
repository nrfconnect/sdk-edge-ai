/*
 * Copyright (c) 2026 Nordic Semiconductor ASA
 *
 * SPDX-License-Identifier: LicenseRef-Nordic-5-Clause
 */

#include "imu.h"

#include <zephyr/types.h>
#include <zephyr/device.h>
#include <zephyr/kernel.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/logging/log.h>

LOG_MODULE_REGISTER(imu, CONFIG_LOG_DEFAULT_LEVEL);

#if IS_ENABLED(CONFIG_IMU_FIFO_BATCH)
#include <zephyr/drivers/sensor_data_types.h>
#include <zephyr/rtio/rtio.h>
#include <zephyr/sys/util_macro.h>
#include <math.h>

#define IMU_RING_SIZE  64
#define DECODE_BATCH   8
#define STREAMDEV_ALIAS DT_ALIAS(stream0)
#define IMU_DEV DEVICE_DT_GET(STREAMDEV_ALIAS)
#else
#define IMU_DEV DEVICE_DT_GET_ONE(bosch_bmi270)
#endif

static struct {
	bool initialized;
	generic_cb_t data_ready_cb;
	const struct device *dev;
#if IS_ENABLED(CONFIG_IMU_FIFO_BATCH)
	const struct sensor_decoder_api *decoder;
	imu_data_t ring[IMU_RING_SIZE];
	volatile uint32_t ring_head;
	volatile uint32_t ring_tail;
#endif
} imu_ctx = {
	.initialized = false,
	.data_ready_cb = NULL,
	.dev = IMU_DEV,
};

static status_t imu_configure_sensor(const imu_config_t *p_config)
{
	int ret;
	struct sensor_value val;

	val.val1 = p_config->accel_fs_g;
	val.val2 = 0;
	ret = sensor_attr_set(imu_ctx.dev, SENSOR_CHAN_ACCEL_XYZ,
			      SENSOR_ATTR_FULL_SCALE, &val);
	HW_RETURN_IF(ret != 0, STATUS_HARDWARE_ERROR);

	val.val1 = 1;
	val.val2 = 0;
	ret = sensor_attr_set(imu_ctx.dev, SENSOR_CHAN_ACCEL_XYZ,
			      SENSOR_ATTR_OVERSAMPLING, &val);
	HW_RETURN_IF(ret != 0, STATUS_HARDWARE_ERROR);

	val.val1 = p_config->data_rate_hz;
	val.val2 = 0;
	ret = sensor_attr_set(imu_ctx.dev, SENSOR_CHAN_ACCEL_XYZ,
			      SENSOR_ATTR_SAMPLING_FREQUENCY, &val);
	HW_RETURN_IF(ret != 0, STATUS_HARDWARE_ERROR);

	val.val1 = p_config->gyro_fs_dps;
	val.val2 = 0;
	ret = sensor_attr_set(imu_ctx.dev, SENSOR_CHAN_GYRO_XYZ,
			      SENSOR_ATTR_FULL_SCALE, &val);
	HW_RETURN_IF(ret != 0, STATUS_HARDWARE_ERROR);

	val.val1 = 1;
	val.val2 = 0;
	ret = sensor_attr_set(imu_ctx.dev, SENSOR_CHAN_GYRO_XYZ,
			      SENSOR_ATTR_OVERSAMPLING, &val);
	HW_RETURN_IF(ret != 0, STATUS_HARDWARE_ERROR);

	val.val1 = p_config->data_rate_hz;
	val.val2 = 0;
	ret = sensor_attr_set(imu_ctx.dev, SENSOR_CHAN_GYRO_XYZ,
			      SENSOR_ATTR_SAMPLING_FREQUENCY, &val);
	HW_RETURN_IF(ret != 0, STATUS_HARDWARE_ERROR);

	return STATUS_SUCCESS;
}

#if IS_ENABLED(CONFIG_IMU_FIFO_BATCH)

#define STREAM_TRIGGERS \
	{ SENSOR_TRIG_FIFO_WATERMARK, SENSOR_STREAM_DATA_INCLUDE }

SENSOR_DT_STREAM_IODEV(stream_iodev, STREAMDEV_ALIAS, STREAM_TRIGGERS);

RTIO_DEFINE_WITH_MEMPOOL(stream_ctx,
			  /* sq_sz */ 1, /* cq_sz */ 1,
			  /* pool_cnt */ 4,
			  /* pool_blk */ 640,
			  /* pool_align */ sizeof(void *));

static struct sensor_chan_spec accel_chan = { SENSOR_CHAN_ACCEL_XYZ, 0 };
static struct sensor_chan_spec gyro_chan  = { SENSOR_CHAN_GYRO_XYZ, 0 };

#define STREAM_STACK_SIZE 4096
static K_THREAD_STACK_DEFINE(stream_stack, STREAM_STACK_SIZE);
static struct k_thread stream_thread_data;

static inline uint32_t ring_next(uint32_t idx)
{
	return (idx + 1) % IMU_RING_SIZE;
}

static uint32_t ring_drops;

static void ring_push(const imu_data_t *sample)
{
	uint32_t next = ring_next(imu_ctx.ring_head);

	if (next == imu_ctx.ring_tail) {
		ring_drops++;
		return;
	}
	imu_ctx.ring[imu_ctx.ring_head] = *sample;
	imu_ctx.ring_head = next;
}

static bool ring_pop(imu_data_t *sample)
{
	if (imu_ctx.ring_head == imu_ctx.ring_tail) {
		return false;
	}
	*sample = imu_ctx.ring[imu_ctx.ring_tail];
	imu_ctx.ring_tail = ring_next(imu_ctx.ring_tail);
	return true;
}

static void decode_and_push(const struct sensor_three_axis_data *acc,
			    const struct sensor_three_axis_data *gyr,
			    float acc_scale, float gyr_scale,
			    int idx)
{
	imu_data_t d;

	for (int a = 0; a < IMU_NUM_AXES; a++) {
		d.accel[a].phys = (float)acc->readings[idx].values[a] * acc_scale;
		d.accel[a].raw = (int16_t)(d.accel[a].phys * 1000.0f);
	}
	for (int g = 0; g < IMU_NUM_AXES; g++) {
		d.gyro[g].phys = (float)gyr->readings[idx].values[g] * gyr_scale;
		d.gyro[g].raw = (int16_t)(d.gyro[g].phys * 1000.0f);
	}
	ring_push(&d);
}

#define THREE_AXIS_BUF_SIZE (sizeof(struct sensor_three_axis_data) + \
			     (DECODE_BATCH - 1) * sizeof(struct sensor_three_axis_sample_data))

static void stream_entry(void *p1, void *p2, void *p3)
{
	ARG_UNUSED(p1);
	ARG_UNUSED(p2);
	ARG_UNUSED(p3);

	uint8_t acc_buf[THREE_AXIS_BUF_SIZE];
	uint8_t gyr_buf[THREE_AXIS_BUF_SIZE];

	struct sensor_three_axis_data *acc_data = (struct sensor_three_axis_data *)acc_buf;
	struct sensor_three_axis_data *gyr_data = (struct sensor_three_axis_data *)gyr_buf;
	struct rtio_sqe *handle;
	int ret;

	sensor_stream(&stream_iodev, &stream_ctx, NULL, &handle);

	for (;;) {
		struct rtio_cqe *cqe;
		uint8_t *buf;
		uint32_t buf_len;
		uint16_t acc_total, gyr_total;

		cqe = rtio_cqe_consume_block(&stream_ctx);
		if (cqe->result < 0) {
			LOG_WRN("stream CQE error: %d", cqe->result);
			rtio_cqe_release(&stream_ctx, cqe);
			continue;
		}

		ret = rtio_cqe_get_mempool_buffer(&stream_ctx, cqe,
						  &buf, &buf_len);
		rtio_cqe_release(&stream_ctx, cqe);
		if (ret != 0) {
			continue;
		}

		ret = imu_ctx.decoder->get_frame_count(buf, accel_chan,
						       &acc_total);
		if (ret != 0) {
			acc_total = 0;
		}
		ret = imu_ctx.decoder->get_frame_count(buf, gyro_chan,
						       &gyr_total);
		if (ret != 0) {
			gyr_total = 0;
		}

		uint16_t count = MIN(acc_total, gyr_total);
		uint32_t acc_decoded = 0, gyr_decoded = 0;
		float acc_scale = 0.0f, gyr_scale = 0.0f;

		while (acc_decoded < count) {
			uint16_t batch_len = MIN(DECODE_BATCH, count - acc_decoded);
			int na = imu_ctx.decoder->decode(buf, accel_chan,
							 &acc_decoded, batch_len,
							 acc_data);
			int ng = imu_ctx.decoder->decode(buf, gyro_chan,
							 &gyr_decoded, batch_len,
							 gyr_data);
			int n = MIN(na, ng);

			if (acc_scale == 0.0f && na > 0) {
				acc_scale = ldexpf(1.0f, acc_data->shift - 31);
				gyr_scale = ldexpf(1.0f, gyr_data->shift - 31);
			}

			for (int k = 0; k < n; k++) {
				decode_and_push(acc_data, gyr_data,
						acc_scale, gyr_scale, k);
			}

			if (na <= 0 || ng <= 0) {
				break;
			}
		}

		rtio_release_buffer(&stream_ctx, buf, buf_len);

		if (ring_drops > 0) {
			LOG_WRN("ring overflow: %u samples dropped", ring_drops);
			ring_drops = 0;
		}

		if (imu_ctx.data_ready_cb) {
			imu_ctx.data_ready_cb();
		}
	}
}

status_t imu_init(const imu_config_t *p_config,
		  generic_cb_t data_ready_cb)
{
	int ret;

	NULL_CHECK(p_config);
	VERIFY_VALID_ARG(p_config->data_rate_hz > 0);
	HW_RETURN_IF(imu_ctx.dev == NULL, STATUS_HARDWARE_ERROR);
	HW_RETURN_IF(!device_is_ready(imu_ctx.dev), STATUS_HARDWARE_ERROR);

	ret = sensor_get_decoder(imu_ctx.dev, &imu_ctx.decoder);
	HW_RETURN_IF(ret != 0, STATUS_HARDWARE_ERROR);

	status_t status = imu_configure_sensor(p_config);

	if (status != STATUS_SUCCESS) {
		return status;
	}

	imu_ctx.data_ready_cb = data_ready_cb;
	imu_ctx.ring_head = 0;
	imu_ctx.ring_tail = 0;

	k_thread_create(&stream_thread_data, stream_stack,
			K_THREAD_STACK_SIZEOF(stream_stack),
			stream_entry, NULL, NULL, NULL,
			K_PRIO_COOP(5), 0, K_NO_WAIT);

	imu_ctx.initialized = true;
	return STATUS_SUCCESS;
}

status_t imu_read(imu_data_t *const p_data)
{
	NULL_CHECK(p_data);
	HW_RETURN_IF(imu_ctx.dev == NULL, STATUS_HARDWARE_ERROR);

	if (!ring_pop(p_data)) {
		return STATUS_UNAVAILABLE;
	}
	return STATUS_SUCCESS;
}

#else /* !CONFIG_IMU_FIFO_BATCH */

static void data_read_timer_handler(struct k_timer *timer)
{
	ARG_UNUSED(timer);
	if (imu_ctx.data_ready_cb) {
		imu_ctx.data_ready_cb();
	}
}

K_TIMER_DEFINE(data_ready_timer, data_read_timer_handler, NULL);

status_t imu_init(const imu_config_t *p_config,
			  generic_cb_t data_ready_cb)
{
	uint32_t data_ready_timer_period;

	NULL_CHECK(p_config);
	VERIFY_VALID_ARG(p_config->data_rate_hz > 0);
	HW_RETURN_IF(imu_ctx.dev == NULL, STATUS_HARDWARE_ERROR);
	HW_RETURN_IF(!device_is_ready(imu_ctx.dev), STATUS_HARDWARE_ERROR);

	status_t status = imu_configure_sensor(p_config);

	if (status != STATUS_SUCCESS) {
		return status;
	}

	imu_ctx.data_ready_cb = data_ready_cb;
	data_ready_timer_period = MAX(1U, (uint32_t)(1000U / p_config->data_rate_hz));
	k_timer_start(&data_ready_timer, K_MSEC(data_ready_timer_period),
		      K_MSEC(data_ready_timer_period));

	return STATUS_SUCCESS;
}

status_t imu_read(imu_data_t *const p_data)
{
	struct sensor_value acc[IMU_NUM_AXES], gyr[IMU_NUM_AXES];
	int res;

	NULL_CHECK(p_data);
	HW_RETURN_IF(imu_ctx.dev == NULL, STATUS_HARDWARE_ERROR);

	res = sensor_sample_fetch(imu_ctx.dev);
	HW_RETURN_IF(res != 0, STATUS_HARDWARE_ERROR);

	res = sensor_channel_get(imu_ctx.dev, SENSOR_CHAN_ACCEL_XYZ, acc);
	HW_RETURN_IF(res != 0, STATUS_HARDWARE_ERROR);

	res = sensor_channel_get(imu_ctx.dev, SENSOR_CHAN_GYRO_XYZ, gyr);
	HW_RETURN_IF(res != 0, STATUS_HARDWARE_ERROR);

	for (int i = 0; i < IMU_NUM_AXES; i++) {
		p_data->accel[i].phys = (float)sensor_value_to_double(&acc[i]);
		p_data->accel[i].raw = (p_data->accel[i].phys * 1000);

		p_data->gyro[i].phys = (float)sensor_value_to_double(&gyr[i]);
		p_data->gyro[i].raw = (p_data->gyro[i].phys * 1000);
	}

	return STATUS_SUCCESS;
}

#endif /* CONFIG_IMU_FIFO_BATCH */
