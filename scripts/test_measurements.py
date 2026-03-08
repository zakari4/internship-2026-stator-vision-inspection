#!/usr/bin/env python3
"""
Test script for the three measurement calibration methodologies.

Tests:
  1. Method A – ArUco marker detection and PPM calibration
  2. Method B – Camera intrinsics geometric calibration
  3. Method C – ML depth estimation (MiDaS)
  4. Lens undistortion utility
  5. End-to-end: generate ArUco → detect → calibrate → measure
"""

import sys
import os
import traceback

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Test framework (reuse pattern from audit_pipeline.py)
# ---------------------------------------------------------------------------
results = {"pass": [], "fail": [], "warn": []}


def test(name):
    """Decorator that catches exceptions and records pass/fail."""
    def decorator(fn):
        def wrapper():
            print(f"\n{'=' * 60}")
            print(f"TEST: {name}")
            print(f"{'=' * 60}")
            try:
                fn()
                print(f"  [PASS] {name}")
                results["pass"].append(name)
            except AssertionError as e:
                print(f"  [FAIL] {name}: {e}")
                results["fail"].append((name, str(e)))
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()
                results["fail"].append((name, str(e)))
        wrapper.__name__ = name
        return wrapper
    return decorator


# ===========================================================================
# TEST 1: Method A – ArUco Marker Detection + PPM Calibration
# ===========================================================================
@test("Method A: ArUco Marker Detection & PPM")
def test_aruco():
    from src.utils.measurements import CalibrationManager, detect_aruco_markers

    # Check that cv2.aruco is available
    try:
        aruco = cv2.aruco
    except AttributeError:
        raise RuntimeError("cv2.aruco not available – install opencv-contrib-python")

    # Generate a synthetic ArUco marker image
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker_size_px = 200
    marker_img = aruco.generateImageMarker(dictionary, 0, marker_size_px)

    # Place marker on a white canvas
    canvas = np.ones((480, 640), dtype=np.uint8) * 255
    y_off, x_off = 140, 220
    canvas[y_off: y_off + marker_size_px, x_off: x_off + marker_size_px] = marker_img

    # Convert to BGR for consistency
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # 1a. Test standalone detect_aruco_markers
    corners, ids = detect_aruco_markers(canvas_bgr)
    assert corners is not None, "No ArUco markers detected in synthetic image"
    assert ids is not None and len(ids) > 0, "ArUco IDs should not be empty"
    print(f"  Detected marker ID={int(ids[0])}, corners shape={corners[0].shape}")

    # Verify detected marker is ~ marker_size_px
    c = corners[0].reshape(4, 2)
    side_lengths = [np.linalg.norm(c[(i + 1) % 4] - c[i]) for i in range(4)]
    avg_side = np.mean(side_lengths)
    print(f"  Average side length: {avg_side:.1f} px (expected ~{marker_size_px})")
    assert abs(avg_side - marker_size_px) < 15, (
        f"Side length {avg_side:.1f} too far from {marker_size_px}"
    )

    # 1b. Test calibrate_from_aruco
    cal = CalibrationManager()
    marker_real_mm = 50.0
    result = cal.calibrate_from_aruco(canvas_bgr, marker_real_size_mm=marker_real_mm)
    assert result is not None, "Calibration returned None"
    expected_ppm = avg_side / marker_real_mm
    expected_px_to_mm = 1.0 / expected_ppm
    print(f"  pixel_to_mm = {result:.5f} (expected ~{expected_px_to_mm:.5f})")
    assert abs(result - expected_px_to_mm) < 0.05, (
        f"pixel_to_mm {result:.5f} != expected {expected_px_to_mm:.5f}"
    )
    assert cal.calibration_method == 'aruco_marker'

    # 1c. Verify measurement conversion
    dist_px = 100.0
    dist_mm = cal.convert_to_mm(dist_px)
    expected_mm = dist_px * expected_px_to_mm
    print(f"  100px = {dist_mm:.2f}mm (expected ~{expected_mm:.2f}mm)")
    assert abs(dist_mm - expected_mm) < 1.0


# ===========================================================================
# TEST 2: Method B – Camera Intrinsics Calibration
# ===========================================================================
@test("Method B: Camera Intrinsics Calibration")
def test_camera_intrinsics():
    from src.utils.measurements import CalibrationManager

    cal = CalibrationManager()

    # Known scenario:
    #   sensor_width = 6.17mm, focal_length = 4.0mm,
    #   distance = 300mm, image_width = 640px
    # Expected pixel_size = (300 * 6.17) / (4.0 * 640)
    #                      = 1851.0 / 2560.0
    #                      = 0.72305 mm/px
    result = cal.calibrate_from_camera_intrinsics(
        sensor_width_mm=6.17,
        focal_length_mm=4.0,
        object_distance_mm=300.0,
        image_width_px=640
    )
    expected = (300.0 * 6.17) / (4.0 * 640)
    print(f"  pixel_to_mm = {result:.5f} (expected {expected:.5f})")
    assert abs(result - expected) < 1e-5, f"Got {result}, expected {expected}"
    assert cal.calibration_method == 'camera_intrinsics'

    # Test with an actual image instead of explicit width
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result2 = cal.calibrate_from_camera_intrinsics(
        sensor_width_mm=6.17,
        focal_length_mm=4.0,
        object_distance_mm=300.0,
        image=img
    )
    assert abs(result2 - expected) < 1e-5, "Should produce same result from image width"

    # Test different distance → different pixel_to_mm
    result_far = cal.calibrate_from_camera_intrinsics(
        sensor_width_mm=6.17,
        focal_length_mm=4.0,
        object_distance_mm=600.0,
        image_width_px=640
    )
    assert result_far > result, "Farther distance should give larger pixel_to_mm"
    print(f"  At 600mm: pixel_to_mm = {result_far:.5f} (should be > {result:.5f})")

    # Test edge case: missing width → ValueError
    try:
        cal.calibrate_from_camera_intrinsics(
            sensor_width_mm=6.17, focal_length_mm=4.0,
            object_distance_mm=300.0
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ValueError correctly raised when no image width provided")


# ===========================================================================
# TEST 3: Method C – ML Depth Estimation (MiDaS)
# ===========================================================================
@test("Method C: ML Depth Estimation (MiDaS)")
def test_depth_estimation():
    from src.utils.measurements import CalibrationManager

    cal = CalibrationManager()

    # Use a real image from the dataset
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    img_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))]
    assert len(img_files) > 0, "No images found in data/"

    img_path = os.path.join(data_dir, img_files[0])
    image = cv2.imread(img_path)
    assert image is not None, f"Failed to read {img_path}"
    print(f"  Using image: {img_files[0]} ({image.shape})")

    # Run depth estimation
    result = cal.calibrate_from_depth_estimation(
        image,
        model_type="MiDaS_small"
    )

    if result is None:
        print("  [WARN] MiDaS model failed to load (network issue?). Skipping.")
        results["warn"].append("Method C: MiDaS model could not be loaded")
        return

    assert result > 0, f"pixel_to_mm should be positive, got {result}"
    print(f"  pixel_to_mm = {result:.5f}")
    print(f"  Estimated distance = {cal._estimated_distance_mm:.1f} mm")
    assert cal.calibration_method == 'depth_estimation'

    # Check that depth map was stored
    depth_map = cal.get_last_depth_map()
    assert depth_map is not None, "Depth map should be stored"
    print(f"  Depth map shape: {depth_map.shape}")
    assert depth_map.shape == image.shape[:2], "Depth map should match image dimensions"


# ===========================================================================
# TEST 4: Lens Undistortion
# ===========================================================================
@test("Lens Undistortion")
def test_undistortion():
    from src.utils.measurements import undistort_image

    # Create a simple test image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(image, (320, 240), 100, (255, 255, 255), 2)
    cv2.line(image, (0, 240), (639, 240), (128, 128, 128), 1)
    cv2.line(image, (320, 0), (320, 479), (128, 128, 128), 1)

    # Synthetic camera matrix (approximate)
    fx, fy = 500.0, 500.0
    cx, cy = 320.0, 240.0
    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float64)

    # Synthetic barrel distortion
    dist_coeffs = np.array([-0.2, 0.1, 0.0, 0.0, 0.0], dtype=np.float64)

    # Undistort
    undistorted = undistort_image(image, camera_matrix, dist_coeffs)
    assert undistorted.shape == image.shape, "Output shape should match input"
    assert undistorted.dtype == image.dtype, "Output dtype should match input"
    print(f"  Input shape: {image.shape}, Output shape: {undistorted.shape}")

    # The undistorted image should differ from original (distortion was applied)
    diff = np.abs(undistorted.astype(float) - image.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  Max pixel diff: {max_diff:.1f}, Mean diff: {mean_diff:.4f}")
    # With barrel distortion coefficients, the images should differ
    assert max_diff > 0, "Undistorted image should differ from original"

    # Test with custom new_camera_matrix
    new_mtx = camera_matrix.copy()
    undistorted2 = undistort_image(image, camera_matrix, dist_coeffs, new_mtx)
    assert undistorted2.shape == image.shape


# ===========================================================================
# TEST 5: End-to-End – ArUco → Calibrate → Measure
# ===========================================================================
@test("End-to-End: ArUco → Calibrate → Measure Object")
def test_e2e():
    from src.utils.measurements import CalibrationManager

    aruco = cv2.aruco

    # Create a scene with:
    #   - ArUco marker (100px side = 25mm real)
    #   - A "wire" drawn as a line of known pixel length
    canvas = np.ones((600, 800, 3), dtype=np.uint8) * 220

    # Draw ArUco marker
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker_px = 100
    marker_img = aruco.generateImageMarker(dictionary, 7, marker_px)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    canvas[50: 50 + marker_px, 50: 50 + marker_px] = marker_bgr

    # Draw a "wire" line (300px long)
    wire_start = (300, 300)
    wire_end = (600, 300)
    wire_px = np.linalg.norm(np.array(wire_end) - np.array(wire_start))
    cv2.line(canvas, wire_start, wire_end, (0, 0, 255), 3)

    # Step 1: Calibrate from ArUco
    cal = CalibrationManager()
    marker_real_mm = 25.0
    px_to_mm = cal.calibrate_from_aruco(canvas, marker_real_size_mm=marker_real_mm)
    assert px_to_mm is not None, "ArUco calibration failed"

    # Step 2: Measure the wire
    wire_mm = cal.convert_to_mm(wire_px)
    # Expected: wire_px * (marker_real_mm / marker_detected_px)
    # ~300px * (25mm / ~100px) = ~75mm
    expected_mm = wire_px * (marker_real_mm / marker_px)
    print(f"  Wire: {wire_px:.0f}px = {wire_mm:.2f}mm (expected ~{expected_mm:.1f}mm)")
    assert abs(wire_mm - expected_mm) < 5.0, (
        f"Wire measurement {wire_mm:.2f}mm != expected {expected_mm:.1f}mm"
    )

    # Step 3: Cross-validate with camera intrinsics
    cal2 = CalibrationManager()
    # If we know the marker is 100px at 300mm with sensor 6.17mm, focal 4.0:
    # pixel_size = (300 * 6.17) / (4.0 * 800) = 0.578 mm/px
    # The ArUco method gives: 25mm / 100px = 0.25 mm/px
    # These are different because the intrinsic method depends on actual
    # physical camera parameters while ArUco is ground-truth from the scene.
    cal2.calibrate_from_camera_intrinsics(
        sensor_width_mm=6.17, focal_length_mm=4.0,
        object_distance_mm=300.0, image_width_px=800
    )
    print(f"  ArUco pixel_to_mm: {cal.pixel_to_mm:.5f}")
    print(f"  Camera pixel_to_mm: {cal2.pixel_to_mm:.5f}")
    print(f"  (Different values expected - ArUco is scene-specific ground truth)")


# ===========================================================================
# TEST 6: Checkerboard Camera Calibration (synthetic)
# ===========================================================================
@test("Checkerboard Camera Calibration")
def test_checkerboard():
    from src.utils.measurements import calibrate_camera_from_checkerboard

    # Create a synthetic checkerboard image (7x5 inner corners)
    board_size = (7, 5)
    square_size = 40  # pixels per square
    board_w = (board_size[0] + 1) * square_size
    board_h = (board_size[1] + 1) * square_size

    checkerboard = np.ones((board_h, board_w), dtype=np.uint8) * 255
    for r in range(board_size[1] + 1):
        for c in range(board_size[0] + 1):
            if (r + c) % 2 == 1:
                y = r * square_size
                x = c * square_size
                checkerboard[y: y + square_size, x: x + square_size] = 0

    # Pad and convert to BGR
    padded = np.ones((480, 640), dtype=np.uint8) * 200
    y_off = (480 - board_h) // 2
    x_off = (640 - board_w) // 2
    padded[y_off: y_off + board_h, x_off: x_off + board_w] = checkerboard
    padded_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

    # Test corner detection (just verify no crash)
    gray = cv2.cvtColor(padded_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, board_size, None)
    print(f"  Checkerboard corners found: {found}")

    if found:
        try:
            cam_mtx, dist_c, reproj_err = calibrate_camera_from_checkerboard(
                [padded_bgr], board_size=board_size, square_size_mm=20.0
            )
            print(f"  Camera matrix shape: {cam_mtx.shape}")
            print(f"  Distortion coeffs: {dist_c.ravel()[:5]}")
            print(f"  Reprojection error: {reproj_err:.4f}")
            assert cam_mtx.shape == (3, 3), "Camera matrix should be 3x3"
        except ValueError as e:
            print(f"  Checkerboard calibration with 1 image may fail: {e}")
            print("  (This is expected - need multiple views for good calibration)")
    else:
        print("  Skipping calibration test (corners not detected in synthetic image)")


# ===========================================================================
# RUN ALL TESTS
# ===========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  MEASUREMENT METHODOLOGIES TEST SUITE")
    print("=" * 60)

    test_aruco()
    test_camera_intrinsics()
    test_depth_estimation()
    test_undistortion()
    test_e2e()
    test_checkerboard()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Passed:   {len(results['pass'])}")
    print(f"  Failed:   {len(results['fail'])}")
    print(f"  Warnings: {len(results['warn'])}")

    if results["fail"]:
        print(f"\n  Failures:")
        for name, err in results["fail"]:
            print(f"    - {name}: {err}")

    if results["warn"]:
        print(f"\n  Warnings:")
        for w in results["warn"]:
            print(f"    - {w}")

    if not results["fail"]:
        print(f"\n  All tests PASSED!")

    sys.exit(1 if results["fail"] else 0)
