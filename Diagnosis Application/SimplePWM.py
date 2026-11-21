"""Hardware drivers: BH1750 light sensor (I2C) and PWM LED, via lgpio / pigpio.

`lgpio` and `pigpio` exist only on the device (Armbian/Raspberry Pi). The imports are guarded so the application
can still run on a development PC; creating the objects then raises a RuntimeError that the caller handles.
"""
import time

try:
    import lgpio
except ImportError:  # development PC
    lgpio = None


class BH1750Sensor:
    """BH1750 light sensor on the I2C bus (continuous high-resolution mode, 1 lx)."""

    def __init__(self, bus=3, address=0x23):
        if lgpio is None:
            raise RuntimeError("The 'lgpio' module is not installed: the BH1750 sensor is not available.")
        self.handle = lgpio.i2c_open(bus, address)
        lgpio.i2c_write_byte(self.handle, 0x01)  # power on
        lgpio.i2c_write_byte(self.handle, 0x10)  # continuous high-resolution mode
        time.sleep(0.1)

    def read_light(self):
        """Light intensity in lux."""
        count, data = lgpio.i2c_read_device(self.handle, 2)
        if count != 2:
            raise IOError("Failed to read data from the BH1750 sensor")
        return ((data[0] << 8) | data[1]) / 1.2

    def close(self):
        lgpio.i2c_close(self.handle)


class LEDPWMController:
    """LED brightness control (hardware PWM via pigpio). Not used by the main GUI."""

    def __init__(self, gpio_pin=12, frequency=1000):
        import pigpio
        self.pi = pigpio.pi()
        self.gpio_pin = gpio_pin
        self.pi.set_mode(gpio_pin, pigpio.OUTPUT)
        self.pi.set_PWM_frequency(gpio_pin, frequency)
        self.pi.set_PWM_range(gpio_pin, 100)

    def set_brightness(self, duty_cycle):
        """Set the brightness, 0-100 %."""
        if not 0 <= duty_cycle <= 100:
            raise ValueError("Duty cycle must be between 0 and 100")
        self.pi.set_PWM_dutycycle(self.gpio_pin, duty_cycle)

    def close(self):
        self.pi.set_PWM_dutycycle(self.gpio_pin, 0)
        self.pi.stop()
