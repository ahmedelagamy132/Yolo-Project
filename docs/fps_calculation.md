# FPS calculation

Ultralytics reports preprocessing, inference, and postprocessing latency in milliseconds. Add those three values to get the total time per image, then calculate:

```text
FPS = 1000 / total latency in milliseconds
```

The benchmark scripts use this formula for their reported FPS.
