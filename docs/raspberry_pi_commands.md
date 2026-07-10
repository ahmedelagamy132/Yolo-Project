# Raspberry Pi monitoring commands

```bash
vcgencmd measure_temp
vcgencmd get_config arm_freq
vcgencmd get_config int
vcgencmd measure_clock arm
vcgencmd measure_volts uncached
```

Typical defaults include `arm_freq=2400` and `over_voltage=0`, although values depend on the board and firmware. The Raspberry Pi's integrated GPU is generally not used by standard Ultralytics/PyTorch inference, so GPU utilization may not be reported.
