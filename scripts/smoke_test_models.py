import subprocess
import json
import os
import sys

MODELS = [
    'unet_lightweight', 'unet_resnet18', 'deeplabv3_mobilenet', 'segformer_b0', 'mask_rcnn',
    'yolov8n_seg', 'yolov8s_seg', 'yolov8m_seg',
    'yolov11n_seg', 'yolov11s_seg', 'yolov11m_seg',
    'yolov26n_seg', 'yolov26s_seg', 'yolov26m_seg',
    'rtdetr_l', 'rtdetr_x', 'codetr', 'rf_detr',
    'sam2_lora', 'internimage', 'setr', 'segnext', 'oneformer', 'mask2former',
    'scyllanet', 'hed', 'rcf', 'bdcn', 'pidinet', 'teed', 'ldc'
]

def run_test(model_name):
    print(f"Testing model: {model_name}...")
    cmd = [
        "./venv/bin/python", "scripts/train.py",
        "--mode", "benchmark",
        "--models", model_name,
        "--warmup", "1",
        "--runs", "1"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"  [PASS] {model_name}")
            return True, ""
        else:
            print(f"  [FAIL] {model_name}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {model_name}")
        return False, "Timeout"
    except Exception as e:
        print(f"  [ERROR] {model_name}: {e}")
        return False, str(e)

def main():
    results = {}
    failed = []
    
    for model in MODELS:
        success, error = run_test(model)
        results[model] = {"success": success, "error": error}
        if not success:
            failed.append(model)
            
    print("\n" + "="*50)
    print("Smoke Test Summary")
    print("="*50)
    print(f"Total models: {len(MODELS)}")
    print(f"Passed: {len(MODELS) - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed Models:")
        for f in failed:
            print(f" - {f}")
            
    with open("outputs/smoke_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
