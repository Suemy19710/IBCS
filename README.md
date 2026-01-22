# Jugo AI IBCS Visual Analyzer
A prototype tool that analyzes dashboards using AI to check if their charts follow IBCS visual standards
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
  <img width="1467" height="1089" alt="image" src="https://github.com/user-attachments/assets/ea8e2aa9-9929-470e-9a8c-00af90eae770" />
2. Choose:
- Upload Dashboard Image, or
- Take Photo
<img width="1622" height="921" alt="image" src="https://github.com/user-attachments/assets/42527a5d-5339-4ec3-97aa-221efe78640d" />

3. Wait for automated analysis
  <img width="514" height="552" alt="image" src="https://github.com/user-attachments/assets/cc2a1af0-730a-441a-bff4-cc11709feaf9" />

5. Review compliance result + recommendations
   
 Example of compliant dashboard result:
<img width="1536" height="896" alt="image" src="https://github.com/user-attachments/assets/2266605d-ea18-4db0-a1dc-b19655343596" />

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
  
## Backend API (Prototype)
Request:
```python
POST /api/predict
```
Accepts:
- multipart/form-data
- field name: file
- formats: .jpg, .png
Example response:
```json
{
  "label": "Non-compliant",
  "rule": "S1_AxisNotZero",
  "confidence": 0.87,
  "feedback": [
    "The axis does not start at zero. IBCS recommends starting value axes at zero."
  ],
  "model_used": "YOLO"
}
```
Or for compliant dashboards:
```json
{
  "label": "Compliant",
  "confidence": 0.93,
  "rule": "Compliant",
  "feedback": [],
  "model_used": "MobileNet"
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

Prototype focuses on:
- Zero-baseline scaling rule
- Basic visual clarity heuristics

## Flowchart Reference
The model decision process can be summarized as:
- MobileNet / YOLO evaluate the dashboard image.
- The system decides Compliant or Non-compliant.
- For suspected non-compliance related to the axis (S1: Axis), YOLO is used to confirm whether the scale starts at zero.
- Final decision + feedback is returned to the frontend.

<img width="838" height="567" alt="image" src="https://github.com/user-attachments/assets/75743c85-7e3b-4ea5-b0c8-e1dfb2e73b44" />

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
