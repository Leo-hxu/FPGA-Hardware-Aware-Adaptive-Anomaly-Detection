# FPGA Adaptive NN Baseline

This project demonstrates adaptive-threshold anomaly detection under queue load.

## 1) Create / use virtual environment

PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3) Run baseline experiment

```powershell
python baseline_adaptive_nn.py
```

Expected output lines include:

- `saved p99_latency.png`
- `saved accuracy.png`

## 4) Deliverables

After running, the project folder should contain:

- `p99_latency.png`
- `accuracy.png`

These two figures are enough for a prototype demo of adaptive latency control.
