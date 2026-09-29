# 🛡️ InsureAI — End-to-End AI Assistant for Insurance

> A 5-module Streamlit application covering ML Regression, ML Classification, ANN,
> CNN (Transfer Learning), LSTM (Sentiment Analysis), and Generative AI — built as
> the GUVI / HCL Capstone Project.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)

website link : https://insure-ai-capstone.streamlit.app/

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Modules](#modules)
- [Results Summary](#results-summary)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)

---

## Project Overview

**InsureAI** is an end-to-end AI assistant designed for the insurance industry.
It integrates five machine learning and deep learning modules into a single,
production-quality Streamlit web application:

| Module | Task | Technique |
|--------|------|-----------|
| **Module 1** | Premium Prediction & Fraud Detection | Classical ML (Linear Regression, Random Forest, XGBoost) |
| **Module 2** | Premium & Fraud (Deep Learning) | ANN (Dense Neural Networks) |
| **Module 3** | Vehicle Damage Detection | CNN with MobileNetV2 Transfer Learning |
| **Module 4** | Customer Review Sentiment | LSTM (Bidirectional RNN) |
| **Module 5** | Insurance Policy Chatbot | Generative AI (Google Gemini) + Prompt Engineering |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Streamlit UI Layer                   │
│  ┌─────────┬──────────┬─────────┬────────┬─────────┐ │
│  │ Premium │  Fraud   │ Damage  │Sentim- │ GenAI   │ │
│  │ Predict │ Detect   │ Detect  │ent     │ Chatbot │ │
│  └────┬────┴────┬─────┴────┬────┴───┬────┴────┬────┘ │
│       │         │          │        │         │       │
│  ┌────▼────┐┌───▼───┐┌────▼───┐┌───▼───┐┌────▼────┐ │
│  │XGBoost/ ││XGBoost││Mobile- ││BiLSTM ││Gemini   │ │
│  │RF/ANN   ││RF/ANN ││NetV2   ││       ││1.5 Flash│ │
│  └────┬────┘└───┬───┘└────┬───┘└───┬───┘└────┬────┘ │
│       │         │          │        │         │       │
│  ┌────▼─────────▼──────────▼────────▼─────────▼────┐ │
│  │     Preprocessors: Scalers, Encoders, Tokenizer │ │
│  └─────────────────────┬───────────────────────────┘ │
│                        │                              │
│  ┌─────────────────────▼───────────────────────────┐ │
│  │  Data: CSV datasets, Car images, FAQ JSON       │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

---

## Modules

### Module 1 — ML: Premium Prediction (Regression)

- **Task:** Predict annual insurance premium from policyholder details (age, sex, BMI, children, smoker, region).
- **Models:** Linear Regression, Random Forest, XGBoost.
- **Best Model:** XGBoost (R² = 0.9294).
- **Dataset:** 50,060 records (`Insurance_Premium_Dataset.xlsx`).

### Module 1 — ML: Fraud Detection (Classification)

- **Task:** Flag fraudulent insurance claims using claim-level features.
- **Models:** Logistic Regression, Random Forest, XGBoost (with SMOTE for class balancing).
- **Best Model:** XGBoost (F1 = 0.6804, ROC-AUC = 0.905).
- **Dataset:** 48,382 records (`Insurance_Fraud_Claims_Dataset.xlsx`).

### Module 2 — ANN (Deep Learning on Tabular Data)

- **Task:** Replicate Module 1 tasks using Artificial Neural Networks.
- **Architecture:** Input → Dense(64) → Dropout → Dense(32) → Dropout → Dense(16) → Output.
- **Premium ANN:** R² = 0.9174, MAE = ₹2,274.92.
- **Fraud ANN:** Accuracy = 85.64%, F1 = 0.6738, ROC-AUC = 0.902.
- **Regularization:** Dropout + EarlyStopping.

### Module 3 — CNN: Vehicle Damage Detection

- **Task:** Classify uploaded vehicle photos as "Damaged" or "Whole/Intact".
- **Architecture:** MobileNetV2 (frozen backbone) + GlobalAveragePooling + Dense head.
- **Data Augmentation:** RandomFlip, RandomRotation, RandomZoom.
- **Validation Accuracy:** 91.96% (target: ≥ 85%).
- **Dataset:** 2,300 images (1,840 training + 460 validation).

### Module 4 — LSTM: Customer Review Sentiment

- **Task:** Classify customer reviews as Positive or Negative.
- **Architecture:** Embedding(5000, 64) → LSTM(64) → Dropout → Dense(32) → Sigmoid.
- **Tokenizer:** Keras Tokenizer with 5,000 vocab, max sequence length 100.
- **Test Accuracy:** 100% (on curated insurance domain reviews).
- **Dataset:** `insurance_reviews.csv`.

### Module 5 — GenAI: Insurance Policy Chatbot

- **Task:** Answer insurance questions using Generative AI with prompt engineering.
- **Backend:** Google Gemini 1.5 Flash API (with deterministic fallback engine).
- **Prompt Techniques:** Zero-shot, Few-shot (FAQ-grounded), Role/Persona, Chain-of-Thought.
- **Features:** Claim summarization, email drafting, document guidance, off-topic guardrails.
- **API Key Handling:** Secure `.env` file, never hardcoded.

---

## Results Summary

### Regression Models (Premium Prediction)

| Model | R² Score | MAE (₹) | RMSE (₹) |
|-------|----------|---------|----------|
| Linear Regression | 0.8982 | 2,516.58 | 3,200.35 |
| Random Forest | 0.9164 | 2,313.67 | 2,899.36 |
| **XGBoost** | **0.9294** | **2,125.05** | **2,664.75** |
| ANN (Keras) | 0.9174 | 2,274.92 | 2,883.00 |

### Classification Models (Fraud Detection)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.8113 | 0.5070 | 0.8073 | 0.6228 | 0.8881 |
| Random Forest | 0.8320 | 0.5461 | 0.7674 | 0.6381 | 0.8914 |
| **XGBoost** | **0.8727** | **0.6600** | **0.7021** | **0.6804** | **0.9050** |
| ANN (Keras) | 0.8564 | 0.5999 | 0.7684 | 0.6738 | 0.9020 |

### Deep Learning Models

| Module | Model | Key Metric | Value | Target Met |
|--------|-------|------------|-------|------------|
| CNN (Damage) | MobileNetV2 | Val. Accuracy | 91.96% | ✅ (≥ 85%) |
| LSTM (Sentiment) | Bidirectional LSTM | Test Accuracy | 100% | ✅ |

---

## Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/InsureAI.git
   cd InsureAI
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set up Gemini API key for the chatbot:**
   ```bash
   # Create a .env file in the project root:
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```
   > The chatbot works without an API key using its built-in fallback engine.

5. **Run the application:**
   ```bash
   cd app
   streamlit run streamlit_app.py
   ```

6. **Open in browser:** Navigate to `http://localhost:8501`

### Retraining Models (optional)

If you want to retrain all models from scratch:
```bash
python train_models.py
```

---

## Usage

1. **Login:** Use any non-empty username and password (demo mode).
2. **Navigate:** Use the left sidebar to switch between the 5 modules.
3. **Module 1 — Premium Prediction:** Enter policyholder details → click "Predict premium".
4. **Module 2 — Fraud Detection:** Enter claim details → click "Analyze claim".
5. **Module 3 — Damage Detection:** Upload a car photo or use a test sample → click "Detect damage".
6. **Module 4 — Review Sentiment:** Type or paste a customer review → click "Analyze sentiment".
7. **Module 5 — Chatbot:** Type a question or use a quick-prompt pill → click "Send".

---

## Project Structure

```
InsureAI/
├── app/
│   └── streamlit_app.py          # Main Streamlit application (5 modules)
├── data/
│   ├── insurance.csv             # Kaggle medical insurance dataset
│   ├── insurance_premium.csv     # Premium dataset (from Excel)
│   ├── insurance_claims.csv      # Fraud claims dataset (from Excel)
│   ├── insurance_reviews.csv     # Customer review sentiment dataset
│   ├── insurance_faq.json        # FAQ knowledge base for chatbot
│   └── car_damage/               # Car damage images
│       ├── training/             # 1,840 images (2 classes)
│       └── validation/           # 460 images (2 classes)
├── models/
│   ├── premium_model.pkl         # Best ML regression model (XGBoost)
│   ├── premium_scaler.pkl        # StandardScaler for premium features
│   ├── premium_encoder.pkl       # OneHotEncoder for region
│   ├── fraud_model.pkl           # Best ML classifier (XGBoost)
│   ├── fraud_scaler.pkl          # StandardScaler for fraud features
│   ├── fraud_encoder.pkl         # OneHotEncoder for categorical features
│   ├── ann_premium_model.keras   # ANN regression model
│   ├── ann_fraud_model.keras     # ANN classification model
│   ├── cnn_damage_model.keras    # MobileNetV2 transfer learning model
│   ├── cnn_classes.json          # CNN class labels and accuracy
│   ├── lstm_sentiment_model.keras# LSTM sentiment model
│   ├── sentiment_tokenizer.pkl   # Keras Tokenizer for LSTM
│   ├── all_metrics_summary.json  # Combined metrics for all models
│   └── *_interactive.pkl         # Streamlit-specific preprocessors
├── notebooks/
│   ├── module1_ml.ipynb          # EDA + ML training (Regression & Classification)
│   ├── module2_ann.ipynb         # ANN training
│   ├── module3_cnn.ipynb         # CNN transfer learning training
│   ├── module4_lstm.ipynb        # LSTM sentiment training
│   └── module5_genai.ipynb       # GenAI prompt engineering demos
├── reports/
│   └── prompt_engineering_report.md  # Prompt techniques documentation
├── assets/
│   └── sample_images/            # Test images for CNN module
├── train_models.py               # Full model training pipeline
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
├── .env                          # API keys (not tracked in git)
└── README.md                     # This file
```

---

## Screenshots

The application features a professional editorial design with:
- Dark navy sidebar with branded navigation
- Cream/warm white content area with card-based layouts
- Rubber-stamp verdict badges (green/red)
- Monospace metric displays with key-value breakdowns

See the `Mock Screenshot/` directory for high-fidelity design references.

---

## Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10+ |
| **ML/DL** | scikit-learn, XGBoost, TensorFlow/Keras, imbalanced-learn (SMOTE) |
| **CNN** | MobileNetV2 (ImageNet pre-trained, transfer learning) |
| **NLP** | Keras Tokenizer, LSTM, Embedding layers |
| **GenAI** | Google Gemini 1.5 Flash API |
| **App** | Streamlit |
| **Data** | Pandas, NumPy, Pillow |
| **Serialization** | joblib, pickle |

---

## Key Learnings

1. **SMOTE** is essential for imbalanced fraud detection — improved F1 from ~0.40 to ~0.68.
2. **Transfer Learning** (MobileNetV2) reaches 92% accuracy with only 1,840 training images.
3. **Prompt Engineering** techniques (role prompting + few-shot grounding) significantly
   improve chatbot accuracy and domain focus.
4. **Caching** (`@st.cache_resource`) prevents model reload on every Streamlit interaction.
5. **Fallback systems** ensure the app works even without internet or API keys.

---

## Author

**GUVI / HCL Capstone Project — InsureAI**

---

*Built with ❤️ using Python, TensorFlow, and Streamlit*
