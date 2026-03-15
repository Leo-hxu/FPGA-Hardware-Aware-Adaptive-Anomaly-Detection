"""
export_fixed_model.py
Export the trained floating-point model as Q8.8 fixed-point weights in JSON format for FPGA inference.

Usage:
    First run train_tiny_mlp.py to generate export/tiny_mlp_export.json
    Then run this script:
        python export_fixed_model.py
    It will generate export/tiny_mlp_fixed.json
"""

import os
import json
import numpy as np


EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "export")
FLOAT_JSON = os.path.join(EXPORT_DIR, "tiny_mlp_export.json")
FIXED_JSON = os.path.join(EXPORT_DIR, "tiny_mlp_fixed.json")

FRAC_BITS = 8  # Q8.8 format


def float_to_fixed(arr, frac_bits=8):
    """Convert a floating-point array to a fixed-point integer array."""
    scale = 1 << frac_bits
    return np.round(np.asarray(arr) * scale).astype(np.int32)


def main():
    if not os.path.exists(FLOAT_JSON):
        print(f"Floating-point model file not found: {FLOAT_JSON}")
        print("Please run train_tiny_mlp.py first to complete training and export")
        return

    with open(FLOAT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    export_data = {
        "frac_bits": FRAC_BITS,
        "scaler_mean_fixed": float_to_fixed(data["scaler_mean"], FRAC_BITS).tolist(),
        "scaler_scale_fixed": float_to_fixed(data["scaler_scale"], FRAC_BITS).tolist(),
        "fc1_weight_fixed": float_to_fixed(data["fc1_weight"], FRAC_BITS).tolist(),
        "fc1_bias_fixed": float_to_fixed(data["fc1_bias"], FRAC_BITS).tolist(),
        "fc2_weight_fixed": float_to_fixed(data["fc2_weight"], FRAC_BITS).tolist(),
        "fc2_bias_fixed": float_to_fixed(data["fc2_bias"], FRAC_BITS).tolist(),
    }

    os.makedirs(EXPORT_DIR, exist_ok=True)
    with open(FIXED_JSON, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    print(f"Fixed-point model exported to: {FIXED_JSON}")
    print(f"Format: Q{16 - FRAC_BITS}.{FRAC_BITS} (frac_bits={FRAC_BITS})")

    # Print a few sample weights for quick verification.
    print("\n--- fc1_weight (first row) ---")
    print("  float:", [f"{v:.4f}" for v in data["fc1_weight"][0]])
    print("  fixed:", export_data["fc1_weight_fixed"][0])

    print("\n--- fc2_weight ---")
    print("  float:", [f"{v:.4f}" for v in data["fc2_weight"][0]])
    print("  fixed:", export_data["fc2_weight_fixed"][0])


if __name__ == "__main__":
    main()
