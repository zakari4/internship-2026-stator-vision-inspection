# Project State Report — Unified Multi-Domain Inspection Platform

**Date**: 2026-04-10  
**Branch**: dev  
**Repository**: github.com/zakari4/internship-2026-stator-vision-inspection.git

> [!IMPORTANT]
> The platform now runs as a unified multi-domain inference system with three domain managers:
> **Stator**, **Chignon**, and **File**.
>
> File-domain infrastructure is fully integrated in the app, but the model lifecycle is still experimental:
> the current YOLO file-domain runs failed to converge, so production file-domain inference is now based on
> **UNet-ResNet18 (.pth)** plus domain-specific post-processing.

---

## 1. Executive Summary

The project has transitioned from a single-domain stator pipeline into a **Unified Multi-Domain Inspection Platform**.

Key outcomes:

- Domain-aware routing is active across API and WebRTC pipelines (`stator`, `chignon`, `file`).
- Domain-specific model managers are separated and independently configurable.
- File-domain post-processing now includes:
  - color classification inside detected file boxes (`blue`, `yellow`, `green`/fallback),
  - left-right placement validation,
  - center-line visualization between validated pair,
  - UI toggle for enabling/disabling file color/layout validation.
- File-domain YOLO training artifacts are preserved for analysis, but currently not used for live inference.

---

## 2. Project Structure (Current)

```
internship-2026-stator-vision-inspection/
├── server/                        # Flask + WebRTC backend
│   ├── server.py                  # Domain-aware routing, REST APIs, signaling
│   └── inference.py               # Stator inference manager
├── client/                        # Browser UI
│   ├── index.html                 # Domain selector + settings panels
│   └── app.js                     # Domain-aware API calls and controls
├── chignon/                       # Chignon domain package
│   ├── inference.py               # ChignonModelManager
│   └── results/                   # Chignon checkpoints/training outputs
├── files/                         # File domain package
│   ├── inference.py               # FileModelManager (ResNet-only active path)
│   └── results/                   # File domain checkpoints + YOLO logs/plots
├── src/                           # Training/evaluation/model code
├── outputs/                       # Global training, plots, and inference logs
└── docs/                          # Documentation and reports
```

---

## 3. Web Application State

### 3.1 Domain Selector

The frontend domain selector now drives backend manager selection end-to-end:

- `domain=stator` -> `ModelManager`
- `domain=chignon` -> `ChignonModelManager`
- `domain=file` -> `FileModelManager`

This applies to:

- `/api/models`
- `/api/select-model`
- `/api/inference-enhancements`
- `/api/detect`
- WebRTC offer/session routing

### 3.2 Domain-Aware Settings Provider

`/api/inference-enhancements` is now domain-scoped and supports per-domain state.

For file domain, the UI includes a dedicated control:

- `enable_file_color_validation`

When disabled, file-domain post-processing skips color/layout validation and message generation.

---

## 4. Domain Status

### 4.1 Stator Domain — Stable

- Primary use case with mature post-processing and measurement logic.
- Supports YOLO and PyTorch model paths.
- Includes measurement overlays, calibration modes, and geometry-based checks.

### 4.2 Chignon Domain — Integrated

- Dedicated manager (`chignon/inference.py`) in production routing.
- Supports domain-scoped model loading and post-processing controls.
- Available from the same unified app and APIs.

### 4.3 File Domain — Integrated Infrastructure, Experimental Model Lifecycle

- Dedicated manager (`files/inference.py`) integrated in API and WebRTC flows.
- Active inference model: **UNet-ResNet18 checkpoint** from `files/results/checkpoints/unet_resnet18/best_model.pth`.
- Added file-domain post-processing:
  - class output normalized to `class_name=file`,
  - dominant color analysis per bbox,
  - placement validation rule and correctness message,
  - center-line drawing for pair interpretation,
  - UI on/off switch for this validation step.

---

## 5. File Domain Model Results and Decision

### 5.1 Observed File-Domain Training Outcome

From file-domain test runs and generated results under `files/results`:

- YOLO variants (`yolov8m_seg`, `yolov11m_seg`) produced poor/empty detection behavior on validation images.
- Reported metrics/log behavior indicated failed convergence for the intended file-object task.
- In practical app tests, YOLO file-domain detections were often empty or non-usable.

### 5.2 Active Strategy

Because YOLO file-domain training failed in this cycle, the project uses:

- **UNet-ResNet18 (.pth)** as the active file-domain inference model.

This decision is implemented in code by making file-domain discovery/loading ResNet-oriented for production inference.

---

## 6. Architectural Updates Implemented

### 6.1 Diagram Source Update

`generate_diagrams.py` now includes:

- updated detection Mermaid flow with domain manager selection and domain-specific post-processing,
- new `multi_domain_mmd` Mermaid diagram for parallel domain manager architecture,
- encoded outputs for `TRAIN`, `DETECT`, and `MULTI_DOMAIN` diagrams.

### 6.2 Documentation Synchronization

Updated artifacts:

- `README.md` (multi-domain and unified inference narrative)
- `PROJECT_MAP.md` (adds `chignon/` and `files/`)
- `docs/reports/project_state.md` (this report)

---

## 7. Current Risks and Next Steps

### Risks

- File-domain color validation quality depends on lighting and detection box purity.
- File-domain still needs a stronger, retrained detector set for robust two-file pair detection in all scenes.

### Recommended Next Steps

1. Build a dedicated file-domain retraining cycle with stricter label QA and class balance checks.
2. Add threshold calibration presets for color validation (factory-light profiles).
3. Expand file-domain benchmark report with per-scenario precision/recall and failure taxonomies.

---

## 8. Status Summary

- Multi-domain platform: **implemented**
- Stator domain: **stable**
- Chignon domain: **integrated**
- File domain infrastructure: **integrated**
- File domain model maturity: **experimental (ResNet fallback active due to YOLO training failure)**
