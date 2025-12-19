#include "bsp_led.h"
#include <zephyr/device.h>
#include <zephyr/drivers/pwm.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

//////////////////////////////////////////////////////////////////////////////

static const struct pwm_dt_spec red_pwm_led_ = PWM_DT_SPEC_GET(DT_ALIAS(pwm_led0));
static const struct pwm_dt_spec green_pwm_led_ = PWM_DT_SPEC_GET(DT_ALIAS(pwm_led1));
static const struct pwm_dt_spec blue_pwm_led_ = PWM_DT_SPEC_GET(DT_ALIAS(pwm_led2));

//////////////////////////////////////////////////////////////////////////////

// not calibrated, fixed for specific board
#define PWM_PERIOD PWM_MSEC(20)

//////////////////////////////////////////////////////////////////////////////

static int init_led_(const struct pwm_dt_spec pwm_led)
{
    int ret = 0;
    if (!device_is_ready(pwm_led.dev))
    {
        printk("PWM LED Init error '%s' device_is_ready()\n", pwm_led.dev->name);
        return ret;
    }

    ret = pwm_set_dt(&pwm_led, PWM_PERIOD, 0);
    if (ret)
    {
        printk("Error %d: failed to set pulse width for %s\n", ret, pwm_led.dev->name);
        return ret;
    }
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

static int set_led_(const struct pwm_dt_spec pwm_led, float brightness)
{
    int ret = 0;
    uint32_t pulse = brightness * PWM_PERIOD;
    ret = pwm_set_pulse_dt(&pwm_led, pulse);
    if (ret < 0)
    {
        printk("LED Init error pwm_set_pulse_dt()\n");
        return ret;
    }
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_init(void)
{
    int ret;
    ret = init_led_(red_pwm_led_);
    BSP_RETURN_IF(ret != 0, ret);

    ret = init_led_(green_pwm_led_);
    BSP_RETURN_IF(ret != 0, ret);

    ret = init_led_(blue_pwm_led_);
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_set_red(float brightness)
{
    pwm_set_pulse_dt(&green_pwm_led_, 0);
    pwm_set_pulse_dt(&blue_pwm_led_, 0);
    return set_led_(red_pwm_led_, brightness);
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_set_green(float brightness)
{
    pwm_set_pulse_dt(&red_pwm_led_, 0);
    pwm_set_pulse_dt(&blue_pwm_led_, 0);
    return set_led_(green_pwm_led_, brightness);
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_set_blue(float brightness)
{
    pwm_set_pulse_dt(&red_pwm_led_, 0);
    pwm_set_pulse_dt(&green_pwm_led_, 0);
    return set_led_(blue_pwm_led_, brightness);
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_set_rgb(float r, float g, float b)
{
    int ret;
    uint32_t red = r * PWM_PERIOD;
    uint32_t green = g * PWM_PERIOD;
    uint32_t blue = b * PWM_PERIOD;
    ret = pwm_set_pulse_dt(&red_pwm_led_, red);
    BSP_RETURN_IF(ret != 0, ret);

    ret = pwm_set_pulse_dt(&green_pwm_led_, green);
    BSP_RETURN_IF(ret != 0, ret);

    ret = pwm_set_pulse_dt(&blue_pwm_led_, blue);
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_off(void)
{
    int ret;
    ret = pwm_set_pulse_dt(&red_pwm_led_, 0);
    BSP_RETURN_IF(ret != 0, ret);

    ret = pwm_set_pulse_dt(&green_pwm_led_, 0);
    BSP_RETURN_IF(ret != 0, ret);

    ret = pwm_set_pulse_dt(&blue_pwm_led_, 0);
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_blink_red(float brightness, int32_t on_ms, int32_t off_ms)
{
    int ret;
    ret = bsp_led_set_red(brightness);
    BSP_RETURN_IF(ret != 0, ret);

    k_msleep(on_ms);

    ret = bsp_led_off();

    k_msleep(off_ms);
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_blink_green(float brightness, int32_t on_ms, int32_t off_ms)
{
    int ret;
    ret = bsp_led_set_green(brightness);
    BSP_RETURN_IF(ret != 0, ret);

    k_msleep(on_ms);

    ret = bsp_led_off();

    k_msleep(off_ms);
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_blink_blue(float brightness, int32_t on_ms, int32_t off_ms)
{
    int ret;
    ret = bsp_led_set_blue(brightness);
    BSP_RETURN_IF(ret != 0, ret);

    k_msleep(on_ms);

    ret = bsp_led_off();

    k_msleep(off_ms);
    return ret;
}

//////////////////////////////////////////////////////////////////////////////

int bsp_led_blink_rgb(float r, float g, float b, int32_t on_ms, int32_t off_ms)
{
    int ret;
    ret = bsp_led_set_rgb(r, g, b);

    BSP_RETURN_IF(ret != 0, ret);

    k_msleep(on_ms);

    ret = bsp_led_off();

    k_msleep(off_ms);
    return ret;
}