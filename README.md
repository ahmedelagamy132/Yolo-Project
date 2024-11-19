# YOLO Model Performance on Microcontrollers

This repository contains the results and methodology of our research comparing the performance of various YOLO models on different microcontroller platforms. The goal is to evaluate the efficiency and feasibility of deploying state-of-the-art object detection algorithms on edge devices.

---

## Overview

### Objectives
- Benchmark multiple YOLO models (YOLOv1 to YOLOv11 and their derivatives) on various microcontroller platforms.
- Analyze performance metrics including:
  - Latency
  - Frame Per Second (FPS)
  - Power consumption
  - Model size
  - Resource utilization (CPU, GPU, RAM)
  - Disk I/O
  - Temperature

### Microcontroller Platforms
The following microcontroller platforms were evaluated:
- NVIDIA Jetson Nano
- Raspberry Pi (models 4 and 5)
- Google Coral Accelerator
- LattePanda
- Orin

---

## Dataset
The benchmarks use standard datasets for object detection, including:
- MS COCO

These datasets were chosen to maintain consistency with existing YOLO model benchmarks.

---

## YOLO Models
The YOLO models tested include:
- YOLOv1, YOLOv2 ,YOLOv3, YOLOv4, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLOv11
- Tiny YOLO variants (YOLOv3-tiny, YOLOv4-tiny, etc.)
- Nano YOLO models

---

## Metrics
The following performance metrics were measured:
1. **Latency**: Time taken to process a single frame.
2. **FPS**: Number of frames processed per second.
3. **Model Size**: Disk size of the YOLO model.
4. **Power Consumption**: Measured using a USB power meter or INA219 sensor.
5. **Resource Utilization**:
   - CPU usage (%)
   - GPU usage (if applicable)
   - RAM usage
6. **Temperature**: Monitored during operation.
7. **Stress Testing**: Evaluated the microcontroller's ability to handle prolonged high-intensity tasks.

---

## Methodology
1. **Environment Setup**:
   - Installed YOLO model dependencies on microcontrollers.
   - Configured power modes for supported devices (e.g., Jetson Nano's 5W and 10W modes).

2. **Testing**:
   - Deployed YOLO models on each microcontroller.
   - Evaluated models with the selected datasets.
   - Recorded the performance metrics using system monitoring tools (e.g., `htop`, `iotop`, `vcgencmd`).

3. **Data Collection**:
   - Automated scripts were used to measure execution time, resource usage, and other metrics.

---

## Results
Results are documented in the attached spreadsheet: **[Results.xlsx](./Results.xlsx)**.

Highlights include:
- Performance trade-offs between model accuracy and latency.
- Comparison of resource utilization across different microcontrollers.
- Analysis of mAP (mean Average Precision) versus latency.

---

## Conclusion
This research provides insights into the practicality of deploying YOLO models on edge devices. It highlights the trade-offs between model complexity, resource constraints, and real-time performance, offering guidance for selecting YOLO models for specific applications.

---