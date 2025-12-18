# Brain Tumor Segmentation with Knowledge Distillation  
**CMPE 258 – Software Program (Code Submission)**

This repository contains the **codebase, training pipelines, evaluation scripts, and demo application** for our brain tumor MRI segmentation project using **knowledge distillation**.  
High-capacity teacher models (**SegResNet, SwinUNETR**) are distilled into lightweight student models (**UNet, FlexibleUNet**) using the **BraTS** dataset.

---

## 1. Submission Overview (Assignment Compliance)

### Code Submission
- ✅ GitHub repository (this repo)
- ✅ Google Drive shared folder (linked in report & Canvas submission)
- [Google Drive Link](https://drive.google.com/drive/folders/14aoovi9-NYnY5aJC6DnBtX0cuFagpLXg?usp=sharing)
  - Contains:
    - All Google Colab notebooks used for training and evaluation
    - All trained model weights
    - Video demo

### Large Files (Datasets & Models)
- **BraTS dataset**: downloaded automatically via MONAI
- **Trained weights**: hosted on Google Drive  
  → must be downloaded and added locally (see instructions below)

---

## 2. Repository Structure

```text
.
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   └── artifacts/        # Place downloaded model weights here
│   │   ├── serve/                # Ray Serve deployments
│   │   └── inference/
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                    # Streamlit frontend
│   └── requirements.txt
│
├── docker-compose.yml
├── docker-compose.dev.yml
├── .devcontainer/
├── requirements.txt
└── README.md
```

## 3. Reproducibility & Report Alignment

- All evaluation tables and figures shown in the report are generated from:

    - Training notebooks (Google Colab)

    - Evaluation pipelines in this repository

- Metrics include:

  - Validation Dice vs Epoch

  - Validation Loss vs Epoch

  - Dice score distributions

- Results produced by this code match the figures and tables in the report.

## 4. Getting Started
Recommended Environment

- Visual Studio Code (VS Code)

- Docker & Docker Compose

- CUDA 12.1+ compatible GPU

Note: The first build may take time due to large ML dependencies. Subsequent runs start quickly.


### Dependency Management Notes

When using the Docker-based setup, all required system and Python dependencies are installed automatically as part of the container build process. This includes downloading and configuring the language model used by the application to generate textual reports.

If Docker is not available, the project may alternatively be installed using a Python virtual environment. In this case, the required Python packages can be installed from `requirements.txt`, and the language model **gemma3:4b-it-qat** must be set up separately using **Ollama**. The Docker-based workflow is strongly recommended to ensure consistent dependency versions and reproducible execution.


## 5. Option 1: Docker + VS Code Dev Containers (Recommended)
Prerequisites

- Docker installed and running

- VS Code installed

- VS Code extensions:

- Dev Containers

    - Docker

    - Steps

Clone the repository:
``
git clone <repository-url>
cd <project-root>
``

Open VS Code in the project root directory.

Open the Command Palette:

Windows: F1

macOS: Shift + Command + P

Select:
```
Dev Containers: Open Folder in Container...
```

Choose the .devcontainer folder.
VS Code will build and attach to the container automatically.

## 6. Option 2: Manual Docker Build
```
docker compose up --build --detach
```

Then attach VS Code to the running container using the Docker extension or Docker Desktop.

7. Option 3: Python Virtual Environment (Not Recommended)
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 8. Model Weights Setup (Required)
Download from Google Drive

In the shared Google Drive folder, locate:
```
all_model_weight/
├── dev_model/
├── flex/
├── segresnet/
└── unet/
```
Copy these folders into:
```
backend/app/models/artifacts/
```

This step is required to run inference and the frontend demo.

## 9. Running the Backend (Inference & KD Models)

Inside the Docker container (in the backend/ dir):

Start Ray
```
ray start --head --dashboard-host=127.0.0.1
```

This starts Ray Serve and the Ray dashboard.

Start the backend service (in backend/ dir)
```
serve run serve_health:health_app_deployment \
  --name health \
  --route-prefix /
```
### 10. Running the Frontend (Streamlit Demo)

In a new terminal inside the container:

cd frontend
streamlit run app.py --logger.level=debug


Access the application at:
```
http://localhost:8501
```
⚠️ CUDA compatibility and dependency alignment are not guaranteed without Docker.


## 11. Training & Evaluation Notebooks

All training and evaluation were performed using Google Colab notebooks, which are included in the shared Google Drive folder.

These notebooks cover:

- Teacher training (SegResNet, SwinUNETR)

- Student training with knowledge distillation

- Validation and metric computation

- Plot generation used in the report

## 12. Models Used
Teacher Models

- SegResNet – CNN-based, efficient, stable training

- SwinUNETR – Transformer-based, global context modeling

Student Models

- UNet

- FlexibleUNet

Knowledge Distillation

- Multi-label sigmoid outputs

- Dice-based supervised segmentation loss

- MSE/L2 distillation loss on sigmoid probabilities

- Sliding window inference for validation

## 13. Notes for Graders

Large artifacts are hosted externally (Google Drive)
