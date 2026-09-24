# CardioCare AI — Cardiovascular Risk Analyzer & Intelligence

CardioCare is an educational and clinical risk-screening web application. It combines an ensemble machine learning risk prediction engine, automated medical report parsing (PDF, TXT, CSV), and an interactive heart-health educational assistant powered by Hugging Face Inference Providers.

---

## 🌟 Overview

CardioCare AI assists users and clinicians in assessing cardiovascular health risks using standard clinical biomarkers (such as resting blood pressure, serum cholesterol, max heart rate, ST depression / Oldpeak, and ST slope morphology).

The application provides:
- **Instant Risk Scoring & Tiers**: Calculates calibrated cardiovascular risk probability scores (Low, Moderate, High, Critical) with a real-time animated SVG gauge.
- **Biomarker Risk Attribution**: Highlights specifically flagged physiological contributors (e.g., Stage 2 BP, High Cholesterol, Exercise-Induced Angina).
- **Personalized Recommendations**: Generates tailored lifestyle, dietary (DASH/Mediterranean), and clinical follow-up guidance.
- **Medical Report Extraction**: Auto-populates clinical parameters from uploaded lab documents (PDF, TXT, CSV).
- **AI Consultation Assistant**: Answers heart-health questions and explains complex ECG metrics using an LLM.

---

## 🚀 Features

- **Multi-Model Soft-Voting Ensemble**: Combines Random Forest, Gradient Boosting, Extra Trees, Tuned KNN, and Logistic Regression with calibrated probability outputs.
- **Interactive Modern Interface**: Built with modern typography, dark mode aesthetics, dynamic risk meters, and 1-click demo profiles.
- **Document Text Parser**: Extracts diagnostic values directly from text-based PDF reports, CSV files, and TXT summaries.
- **Serverless Ready**: Fully configured for one-click deployment on Vercel and scalable local execution via FastAPI and Uvicorn.
- **Secure Secret Management**: All API keys and tokens are strictly isolated on the backend server and excluded from source control.

---

## 🛠️ Tech Stack

- **Backend / API**: Python 3.10+, FastAPI, Starlette, Pydantic, Uvicorn
- **Frontend**: Vanilla HTML5, Modern CSS (Glassmorphism, CSS Variables), JavaScript (ES6+)
- **Machine Learning**: Scikit-learn, Joblib, NumPy, Pandas, SciPy
- **AI / LLM Integration**: Hugging Face Serverless Inference API (via OpenAI SDK client)
- **Document Processing**: PyPDF, Regex entity parser
- **Deployment**: Vercel Serverless Functions (`@vercel/python`)

---

## 📐 Project Architecture

```
User Browser / Client
    │
    ├──> GET  /              ──> Serves static/index.html (Diagnostic UI & Chat)
    ├──> POST /api/analyze   ──> Normalizes inputs (StandardScaler) & executes Ensemble Model
    ├──> POST /api/report    ──> Parses PDF/TXT/CSV report and extracts clinical values
    └──> POST /api/chat      ──> Communicates with Hugging Face Inference API
```

---

## 🔬 Machine Learning Pipeline

- **Target**: Presence of cardiovascular disease pattern (`1` = Elevated Risk, `0` = Normal Risk).
- **Feature Space (15 features)**: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak, Sex (M/F), ChestPainType (ASY, ATA, NAP, TA), RestingECG (Normal, ST, LVH), ExerciseAngina (Y/N), ST_Slope (Up, Flat, Down).
- **Model Storage**: 
  - `knn_heart_model.pkl`: Soft-voting classifier ensemble.
  - `heart_scaler.pkl`: Fitted `StandardScaler` for continuous feature normalization.
- **Validation**: Evaluated via 5-Fold Stratified Cross-Validation on authentic clinical data, achieving **86.24% Accuracy** and **91.81% ROC-AUC**.

---

## 📁 Project Structure

```
cordio_vascular_risk_analyzer/
├── api/
│   └── index.py             # Vercel serverless function entrypoint
├── static/
│   └── index.html           # Single-page UI, risk gauge, & chat interface
├── app.py                   # FastAPI application & API route handlers
├── heart_disease.py         # Local Uvicorn runner script
├── train_improved_model.py  # Model training & serialization script
├── knn_heart_model.pkl      # Trained soft-voting ensemble model
├── heart_scaler.pkl         # Fitted StandardScaler
├── requirements.txt         # Production Python dependencies
├── vercel.json              # Vercel deployment & routing configuration
├── .env.example             # Environment variables template
├── .gitignore               # Git exclusion rules
└── README.md                # Project documentation
```

---

## 💻 Local Installation & Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Clone the Repository
```bash
git clone <repository-url>
cd cordio_vascular_risk_analyzer/cordio_vascular_risk_analyzer
```

### 3. Create & Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy the example configuration file:
```bash
cp .env.example .env
```
Open `.env` and add your Hugging Face user access token (optional, needed for AI chat):
```env
HF_TOKEN="your_huggingface_token_here"
HF_MODEL="openai/gpt-oss-120b:fastest"
```

### 6. Run the Server
```bash
python heart_disease.py
```
Open your browser at **`http://127.0.0.1:8000`**.

---

## ☁️ Vercel Deployment Guide

This project is configured for deployment on **Vercel** via Python Serverless Functions.

### Step 1: Push Repository to GitHub
```bash
git add .
git commit -m "Prepare CardioCare for Vercel deployment"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Vercel
1. Log in to [Vercel](https://vercel.com).
2. Click **Add New...** → **Project**.
3. Import your GitHub repository.
4. **Root Directory**:
   - If your repository contains the root structure, leave it as `./` (or select `cordio_vascular_risk_analyzer` if deploying from the subfolder).
5. **Environment Variables**:
   - Add `HF_TOKEN` with your Hugging Face access token.
   - (Optional) Add `HF_MODEL` (default: `openai/gpt-oss-120b:fastest`).
6. Click **Deploy**.

---

## 🔗 Live Demo & Links

- **Live Demo**: `[Add Vercel URL after deployment]`
- **Repository**: `[Add GitHub URL after push]`

---

## 📖 API Reference

### 1. Predict Cardiovascular Risk
- **Endpoint**: `POST /api/analyze`
- **Request Body**:
```json
{
  "age": 54,
  "sex": "M",
  "chest_pain": "ASY",
  "resting_bp": 135,
  "cholesterol": 240,
  "fasting_bs": 0,
  "ecg": "Normal",
  "max_hr": 142,
  "angina": "Y",
  "oldpeak": 1.5,
  "st_slope": "Flat"
}
```
- **Response**:
```json
{
  "elevated_pattern": true,
  "risk_score": 53.9,
  "risk_level": "Moderate",
  "key_factors": [
    "High Serum Cholesterol (240 mg/dL)",
    "Exercise-Induced Angina reported during exertion",
    "Significant ST Depression (1.5 mm Oldpeak)"
  ],
  "recommendations": [
    "Adopt a Mediterranean/DASH diet low in saturated fats.",
    "Schedule a cardiac stress test under clinical supervision."
  ],
  "accuracy_benchmark": "86.2% Accuracy / 91.8% ROC-AUC (Multi-Model Ensemble)"
}
```

### 2. Extract Data from Medical Report
- **Endpoint**: `POST /api/report`
- **Format**: `multipart/form-data` with `file` (.pdf, .txt, or .csv)
- **Response**: Extracted values mapped to patient fields.

### 3. AI Heart Health Chat
- **Endpoint**: `POST /api/chat`
- **Request Body**:
```json
{
  "messages": [
    {"role": "user", "content": "What does a Flat ST slope indicate?"}
  ]
}
```
- **Response**: `{"answer": "..."}`

---

## ⚠️ Medical Disclaimer

CardioCare AI is developed strictly for **educational, exploratory, and risk-screening demonstration purposes**. It is **not** a certified medical diagnostic device and cannot replace professional clinical judgment. In case of emergency or severe cardiovascular symptoms (chest pain, shortness of breath, cold sweat), contact your local emergency response number immediately.
