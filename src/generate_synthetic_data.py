"""
generate_synthetic_data.py
Generate synthetic fan current data so the full training pipeline can be validated without real hardware data.
"""

import os
import numpy as np
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def generate_normal_signal(length=2000):
    t = np.arange(length)
    signal = 0.12 + 0.01 * np.sin(2 * np.pi * t / 40) + 0.003 * np.random.randn(length)
    return signal.astype(np.float32)


def generate_blocked_signal(length=2000):
    t = np.arange(length)
    signal = 0.12 + 0.01 * np.sin(2 * np.pi * t / 40) + 0.003 * np.random.randn(length)

    # Add anomalous pulses and fluctuations.
    for _ in range(10):
        start = np.random.randint(0, length - 20)
        signal[start:start + 20] += 0.03 * np.random.rand()

    return signal.astype(np.float32)


def generate_startup_signal(length=2000):
    t = np.arange(length)
    ramp = np.minimum(t / 200.0, 1.0)
    signal = 0.05 + ramp * 0.07 + 0.008 * np.sin(2 * np.pi * t / 25) + 0.004 * np.random.randn(length)
    return signal.astype(np.float32)


def generate_disturb_signal(length=2000):
    t = np.arange(length)
    signal = 0.12 + 0.01 * np.sin(2 * np.pi * t / 40) + 0.003 * np.random.randn(length)

    # Inject a power-supply disturbance in the middle section.
    start = 700
    end = 1000
    signal[start:end] *= 1.25

    return signal.astype(np.float32)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    np.random.seed(42)

    for i in range(3):
        pd.DataFrame({"current": generate_normal_signal()}).to_csv(
            os.path.join(DATA_DIR, f"normal_{i:02d}.csv"), index=False
        )

    for i in range(2):
        pd.DataFrame({"current": generate_blocked_signal()}).to_csv(
            os.path.join(DATA_DIR, f"blocked_{i:02d}.csv"), index=False
        )

    for i in range(2):
        pd.DataFrame({"current": generate_startup_signal()}).to_csv(
            os.path.join(DATA_DIR, f"startup_{i:02d}.csv"), index=False
        )

    for i in range(2):
        pd.DataFrame({"current": generate_disturb_signal()}).to_csv(
            os.path.join(DATA_DIR, f"disturb_{i:02d}.csv"), index=False
        )

    print(f"Synthetic data generated in {DATA_DIR}")


if __name__ == "__main__":
    main()
