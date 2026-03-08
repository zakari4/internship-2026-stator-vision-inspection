#!/usr/bin/env python3
"""
Verification script for dynamic calibration system.
Tests both static and prediction-based calibration logic.
"""

import os
import sys
import numpy as np
import cv2
import torch

# Ensure project root is on sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import config
from src.utils.measurements import CalibrationManager, MeasurementComputer
from src.utils.contour import ContourExtractor
from src.evaluation.metrics import MetricsComputer

def test_static_calibration():
    print("\n[Test] Static Calibration")
    cm = CalibrationManager()
    
    # Initial state (should be from config or 1.0)
    print(f"  Initial PPM: {cm.get_ppm():.4f}")
    
    # Set manually
    cm.set_ppm(10.0)
    print(f"  Manual PPM: {cm.get_ppm():.4f}")
    assert cm.get_ppm() == 10.0
    
    # Reset to default
    cm.reset()
    print(f"  Reset PPM: {cm.get_ppm():.4f}")
    assert abs(cm.get_ppm() - 1.0/config.measurement.pixel_to_mm) < 1e-4

def test_prediction_calibration():
    print("\n[Test] Prediction-based Dynamic Calibration")
    cm = CalibrationManager()
    
    # 1. Test calibration from a mask (integer labeled)
    # mecparts is index 2 in src.data.dataset.LABELS
    from src.data.dataset import LABELS
    ref_name = config.measurement.reference_label_name # 'mecparts'
    ref_idx = LABELS.index(ref_name)
    ref_dim = config.measurement.reference_known_dimension_mm # e.g. 52.0
    
    print(f"  Target Reference: {ref_name} (Index {ref_idx}, {ref_dim}mm)")
    
    # Create mask with ref object at index 2 (CIRCLE with 100px diameter)
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.circle(mask, (250, 250), 50, int(ref_idx), -1)
    
    # Calibrate from mask
    success = cm.calibrate_from_predictions(mask)
    
    if success:
        new_ppm = cm.get_ppm()
        # 100 pixels / 52 mm = 1.923 PPM
        expected_ppm = 100.0 / ref_mm if (ref_mm := ref_dim) > 0 else 1.0
        print(f"  Dynamic PPM: {new_ppm:.4f}")
        print(f"  Expected PPM: {expected_ppm:.4f}")
        assert abs(new_ppm - expected_ppm) < 1e-4
    else:
        print("  Error: Calibration from mask failed")
        assert False

    # 2. Test calibration from a dictionary of contours
    mask_binary = (mask == ref_idx).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_dict = {ref_name: contours}
    
    cm.reset()
    success = cm.calibrate_from_predictions(pred_dict)
    assert success is not None
    assert abs(cm.get_ppm() - expected_ppm) < 1e-4
    print("  Dynamic calibration from Dict passed")

def test_measurement_computation():
    print("\n[Test] Measurement Computation")
    cm = CalibrationManager()
    cm.set_ppm(2.0) # 2 pixels per mm
    
    computer = MeasurementComputer(pixel_to_mm=cm.get_conversion_factor())
    extractor = ContourExtractor()
    
    # Create a 200x100 rectangle (pixels)
    # Should be 100x50 mm
    mask = np.zeros((500, 500), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (250, 150), 255, -1)
    
    contours = extractor.extract_opencv(mask).contours
    results = computer.compute_all_measurements(contours)
    
    # Check diameters (outer_diameter should be (~200px)
    diameters = results.get('diameters', [])
    for i, m in enumerate(diameters):
        print(f"  Measurement {i} ({m.measurement_type}): {m.value_mm:.1f} mm")
        # For a 200x100 rect, minEnclosingCircle diameter is sqrt(200^2+100^2) = 223.6
        # At 2.0 PPM, that's 111.8 mm
        if m.measurement_type == 'outer_diameter':
            assert abs(m.value_mm - 111.8) < 2.0
            print(f"  Outer diameter check passed: {m.value_mm:.1f} mm")

if __name__ == "__main__":
    print("Running Calibration Verification Suite...")
    try:
        test_static_calibration()
        test_prediction_calibration()
        test_measurement_computation()
        print("\nVerification PASSED!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
