# Jugo AI IBCS Visual Analyzer
## Overview

The Jugo AI IBCS Visual Analyzer is a browser-based prototype that uses computer vision to evaluate dashboard images for compliance with the International Business Communication Standards (IBCS). The system automates compliance checks that would otherwise require manual review by analysts or consultants, helping users produce clearer and more consistent dashboards for data-driven communication.

The current prototype focuses on detecting whether chart scales start at zero, demonstrating feasibility for broader automated IBCS evaluation.

## Context and Motivation

Organizations increasingly rely on dashboards to support decision-making. However, inconsistencies in scale, formatting, and visual communication can lead to misinterpretation, errors, and reduced trust in data.

IBCS provides a framework for standardising dashboards, but compliance checks are often:
- Time-consuming
- Expert-dependent
- Manual and subjective

This system bridges the gap between IBCS guidelines and practical adoption by automating visual inspections through AI.


## Project Goal

The goal is to develop an AI-enabled prototype that:
- Accepts dashboard screenshots or camera captures
- Identifies compliance/non-compliance based on visual analysis
- Provides clear and constructive feedback
- Accelerates compliance checks
- Encourages standardisation and best practices


## Key Features

| Feature                       | Description                                                 |
| ----------------------------- | ----------------------------------------------------------- |
| **Dashboard Image Upload** | Upload screenshots or BI exports for instant analysis       |
| **Camera Capture Mode**    | Photograph dashboards from screens or devices               |
| **AI Visual Inspection**   | CNN model evaluates visual layout and structure             |
| **IBCS Rule Checking**     | Prototype currently implements "Start at Zero" scaling rule |
| **Compliance Feedback**    | Clear labeling: Compliant / Non-Compliant                   |
| **Educational Guidance**   | Links to official IBCS resources for improvement            |
| **Fast Inference**         | Results returned in seconds                                 |
| **Web-Based UI**           | No installation or signup required                          |
| **Analyst-Friendly UX**    | Designed for BI teams, consultants, and decision makers     |

## Getting Started
No installation required — Jugo runs entirely in the browser.
Supported Actions:
1. Click Start Your Journey
2. Choose:
- Upload Dashboard Image, or
- Take Photo
3. Wait for automated analysis
4. Review compliance result + recommendations

## Input Requirements
Jugo accepts static dashboard images:
- Formats: JPG, PNG
- Sources:
  - Power BI
  - Tableau
  - Qlik
  - Excel charts
  - Custom reporting tools
  - Photographs of screens
- For best performance:
  - Ensure the full dashboard is visible
  - Avoid glare if using camera
  - Prefer high-contrast images
  
## Example Output (User Perspective)

After processing, Jugo returns one of two outcomes:
```json
IBCS Compliant
Dashboard aligns with core scaling rules and visual standards.
```
or
```json
IBCS Non-Compliant
Dashboard violates IBCS scaling or lacks structural clarity.
```
With recommendations:
```json
“Scale does not start at zero. Consider adjusting the y-axis to a zero baseline according to IBCS proportions.”
```
## Backend API (Prototype)
Request:
```python
POST /api/predict
```
Payload:
```python
file=<image>
```
Response:
```python
{
  "compliance": "Non-Compliant",
  "message": "Scale does not begin at zero. Consider adjusting the axis baseline."
}
```
## AI System Architecture

Jugo’s analysis pipeline consists of three main modules:

1. Image Preprocessing
- RGB conversion
- Resize + normalize
- Noise reduction for camera inputs

2. CNN Model
- Model: MobileNet-based classifier
- Output: Visual compliance probability score
Example
```python
score = model.predict(image)
```
3. IBCS Rule Engine
Threshold logic:
```python
if score >= THRESHOLD:
    label = "Compliant"
else:
    label = "Non-Compliant"
```
Prototype focuses on:
- Zero-baseline scaling rule
- Basic visual clarity heuristics

### IBCS Domain Knowledge
Rule Modeled in Prototype:
| Rule               | Description                                                           |
| ------------------ | --------------------------------------------------------------------- |
| **Starts at Zero** | Charts should begin at zero baseline to preserve proportional meaning |


Future rules planned:


| Rule              | Status  |
| ----------------- | ------- |
| Consistent Scales | Planned |
| Scale Breaks      | Planned |
| Labeled Units     | Planned |
| Misleading Axes   | Planned |

## UX Design Considerations
Jugo supports two key interaction flows:

1. BI Analyst Workflow
- Drag-and-drop
- Quick evaluation
- Instant feedback loop
- Educational reference

2. Consultant & Audit Workflow
- Repeatability
- Evidence-based critique
- Standards education
- Reduced manual visual inspection

## Technical Stack
| Component  | Technology              |
| ---------- | ----------------------- |
| Frontend   | HTML, CSS, JavaScript   |
| Backend    | Python + FastAPI        |
| Model      | MobileNet CNN           |
| Data Flow  | REST API                |
| Deployment | Container / Cloud-ready |
| Platform   | Browser-based UI        |

## Limitations (Current Prototype)
- Visual-only (no data extraction/OCR)
- Single-rule compliance (v1)
- No persistent storage (stateless)
- Sensitive to poor camera inputs
- No user authentication

## Future Work

Short-term roadmap:
- Add multi-rule compliance evaluation
- Support history & comparison view
- Add secure login and user accounts
- Collect labelled dataset for rule training
- Improve visual explainability

Long-term vision:
- Full IBCS rule coverage
- BI platform plugin integration
- Enterprise dashboard auditing workflow

## Security Notes
Current state:
- No data stored
- No authentication required
- Stateless backend
Future state:
- Role-based access control
- Dashboard audit logs
- GDPR alignment for BI artefacts

## Target Users
Designed for:
- BI developers
- Data analysts
- Management consultants
- Business managers
- Corporate reporting teams
- Dashboard design educators

## Authors

Group Jugo
Fontys University of Applied Sciences
2026
