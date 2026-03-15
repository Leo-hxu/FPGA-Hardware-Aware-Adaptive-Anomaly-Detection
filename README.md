# A Runtime-Adaptive FPGA Anomaly Detector for Low-Voltage BLDC Cooling Module Current Waveforms with Power Telemetry

This repository is the software reference implementation for a low-cost FPGA anomaly detection project that monitors low-voltage BLDC cooling module current waveforms and adapts its runtime behavior based on system stress.

The end goal is an FPGA + sensing-board system that performs streaming anomaly detection in real time. The current repository focuses on the software pipeline: data preparation, sliding-window feature extraction, tiny-MLP training, and export of floating-point and fixed-point model parameters for deployment.

## Portfolio Summary

This project is designed to demonstrate the kind of work I want to do in FPGA, embedded AI, and hardware-software co-design roles.

In this repository, I built a compact anomaly detection pipeline for low-voltage BLDC cooling module current waveforms, organized the dataset by physical fault scenario, engineered a lightweight 8-feature representation, trained a deployable tiny MLP, and exported both floating-point and fixed-point model parameters for future FPGA inference.

If you are reviewing this project as a recruiter, hiring manager, or interviewer, the main value is not just the classifier itself. The value is that the project is framed around deployment constraints: latency, power awareness, model simplicity, fixed-point export, and a realistic path from Python reference model to FPGA implementation.

## What This Project Demonstrates

- End-to-end ML pipeline design for a hardware-oriented use case
- Time-series feature engineering for anomaly detection
- Practical model sizing for resource-constrained deployment
- Fixed-point preparation for FPGA inference
- System-level thinking about latency, power, and runtime adaptation
- Clear decomposition from sensing pipeline to deployable inference stages

## Skills Demonstrated

- Python
- NumPy and pandas
- scikit-learn preprocessing and evaluation
- PyTorch model training
- Time-series windowing and feature extraction
- Fixed-point quantization workflow
- Embedded and FPGA deployment thinking
- Hardware-software co-design documentation

## Project Motivation

Real-time anomaly detection on embedded and FPGA platforms is constrained by three competing requirements:

- Detection quality must remain useful under changing operating conditions.
- Tail latency must stay bounded when workload increases.
- Power and hardware cost must remain practical for a small system.

Most anomaly detectors assume a static decision threshold. In practice, queue depth, compute pressure, and power telemetry vary over time. A fixed policy is therefore a poor fit for a real deployment.

This project studies a more deployment-oriented alternative:

`score > f(system_state)`

instead of a fixed rule such as:

`score > T`

The intended adaptive signals are queue depth and power telemetry. The intended adaptive actions are feature-mode switching and threshold switching.

## Final Project Definition

Title:

**A Runtime-Adaptive FPGA Anomaly Detector for Low-Voltage BLDC Cooling Module Current Waveforms with Power Telemetry**

In one sentence:

We monitor the current waveform of a low-voltage BLDC cooling module, detect abnormal behavior with an 8 -> 8 -> 1 tiny MLP, and target an FPGA runtime controller that switches feature complexity and decision threshold according to queue depth and power telemetry.

## Target System

Physical anomalies of interest include:

- Normal spinning
- Partial blockage or friction increase
- Startup and stop transients
- Power disturbance events

Target hardware pipeline:

`BLDC cooling module -> current/power sensor -> FPGA acquisition -> window buffer -> feature extractor -> tiny MLP -> adaptive controller -> anomaly decision`

Planned hardware context:

- Main platform: DE10-Lite
- Sensor front-end: INA219-based current and power telemetry
- Deployment style: streaming inference with lightweight features and fixed-point weights

## Current Repository Scope

This repository contains the software side of the project, not the final FPGA RTL implementation.

What is already implemented:

- Sliding-window segmentation with `window_size = 32` and `stride = 16`
- Eight handcrafted features for each window
- Binary anomaly classification with a tiny MLP
- Model export to floating-point JSON
- Fixed-point export for FPGA-oriented inference
- Synthetic and captured waveform datasets organized by scenario

What this repository is for:

- Golden-model development
- Model bring-up and training
- Feature validation
- Fixed-point handoff preparation for FPGA deployment

## My Contribution

The repository is structured to show a clean engineering path from raw waveform data to deployable model artifacts.

Concretely, I implemented:

- Scenario-based dataset organization for normal, blocked, startup, and disturbance waveforms
- A reusable sliding-window and feature extraction utility layer
- A complete training script for the 8 -> 8 -> 1 tiny MLP baseline
- Export of trained parameters to JSON for software and hardware handoff
- A fixed-point conversion flow suitable for later FPGA integration

This is intentionally a small-model project with a strong deployment story rather than a large-model benchmark project.

## Feature Set

Each window is converted into 8 low-cost features:

1. Mean current
2. Max current
3. Min current
4. Peak-to-peak amplitude
5. Mean absolute difference
6. Window energy
7. Variance
8. Slope or trend

These were chosen because they are cheap to compute, interpretable, and realistic for an FPGA feature extractor.

## Model

The classifier is intentionally small:

- Input: 8
- Hidden: 8 with ReLU
- Output: 1 logit

Architecture:

`8 -> 8 -> 1`

This keeps the model easy to train in software and practical to map into fixed-point FPGA inference.

## Adaptive Runtime Idea

The final hardware-oriented design uses two adaptive knobs:

1. Threshold adaptation

- Low stress: `T_low`
- Medium stress: `T_mid`
- High stress: `T_high`

2. Feature-mode adaptation

- Lite mode: a reduced feature subset for lower runtime cost
- Rich mode: the full 8-feature path for better detection quality

The software code in this repository mainly establishes the feature and model baseline needed before that runtime controller is moved into hardware.

## Repository Structure

```text
data/
  normal_*.csv
  blocked_*.csv
  startup_*.csv
  disturb_*.csv

export/
  tiny_mlp_export.json
  tiny_mlp_fixed.json

src/
  feature_utils.py
  generate_synthetic_data.py
  train_tiny_mlp.py
  export_fixed_model.py
```

## Quick Start

Create an environment and install dependencies:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Generate synthetic data if needed:

```powershell
python src/generate_synthetic_data.py
```

Train the tiny MLP and export the floating-point model:

```powershell
python src/train_tiny_mlp.py
```

Export the model to fixed-point JSON for FPGA-oriented deployment:

```powershell
python src/export_fixed_model.py
```

## Outputs

Training produces:

- Classification metrics on the held-out test split
- `export/tiny_mlp_export.json`
- `export/tiny_mlp_fixed.json`

The fixed-point export is intended to support a future FPGA inference block.

## Why This Is Resume-Relevant

This project is relevant to roles in:

- FPGA and digital design
- embedded systems
- edge AI or tiny ML
- hardware acceleration
- architecture and systems research

The strongest portfolio signal here is that the work connects model design to deployment constraints. Instead of optimizing only for offline accuracy, the project is built around a realistic engineering tradeoff: maintain usable anomaly detection while preparing for bounded latency and low-cost hardware execution.

## Baselines and Evaluation Direction

The broader project compares static and adaptive operating modes:

- Static-Lite
- Static-Rich
- Adaptive-Threshold Only
- Adaptive-Feature + Adaptive-Threshold

The full project evaluation is intended to report:

- Accuracy
- Precision
- Recall
- F1-score
- Average latency
- P95 and P99 latency
- Power or energy per window
- FPGA resource usage such as LUT, FF, BRAM, DSP, and Fmax

## Why This Project Is Interesting

This is not just another anomaly classifier. The interesting part is the system view: the detector is meant to respond to runtime pressure instead of pretending that hardware conditions are static.

That makes the project relevant to:

- FPGA deployment
- embedded ML
- real-time systems
- architecture-aware inference
- low-cost sensing and hardware-software co-design

## Status

Current status:

- Software training pipeline: implemented
- Feature extraction baseline: implemented
- Fixed-point export flow: implemented
- Dataset organization for BLDC current anomaly scenarios: implemented
- FPGA runtime-adaptive deployment: target next stage

Next technical milestone:

- Implement the adaptive controller on FPGA and compare static versus adaptive modes using latency, power, and classification metrics

## Summary

If you are reviewing this project from a portfolio perspective, the core contribution is:

> Built a deployment-oriented anomaly detection pipeline for low-voltage BLDC cooling module current waveforms, including sliding-window feature extraction, 8-feature tiny-MLP training, and fixed-point model export for future FPGA inference. The project is designed around runtime adaptation using queue depth and power telemetry to improve the accuracy-latency-energy tradeoff under changing system load.

Short version suitable for a resume bullet:

> Developed a hardware-oriented anomaly detection pipeline for low-voltage BLDC cooling module current waveforms, including time-series feature extraction, tiny-MLP training, and fixed-point model export for FPGA deployment.
