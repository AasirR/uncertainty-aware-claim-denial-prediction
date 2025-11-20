# Uncertainty-Aware Claim Denial Prediction for Healthcare RCM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Project Overview

An AI-powered framework for predicting healthcare insurance claim denials with built-in uncertainty quantification and human-in-the-loop decision support. This project addresses a critical gap in Healthcare Revenue Cycle Management (RCM) by enabling ML models to **know when they don't know** — allowing them to defer uncertain predictions to human experts.

### Why This Matters

- **15-30%** of healthcare claims are initially denied
- Current ML models provide deterministic predictions without confidence measures
- Incorrect predictions can lead to unexpected patient bills and delayed care
- This framework reduces risk while maintaining operational efficiency

## 🔬 Key Features

- **Uncertainty Quantification**: Monte Carlo Dropout for reliable confidence estimation
- **ML with Rejection**: Models can abstain and defer to human experts when uncertain
- **Safety-Critical AI**: Designed for high-stakes healthcare financial decisions
- **Synthetic Data Generator**: Create realistic RCM datasets for research
- **Human-in-the-Loop Workflow**: Hybrid automation balancing efficiency and safety

## 📊 Dataset

### Synthetic Healthcare Claims Generator

We've developed a realistic synthetic data generator that creates healthcare claims with:

- **Patient demographics** (age ranges)
- **Medical codes** (CPT/ICD-10)
- **Payer information** (Medicare, Medicaid, commercial insurers)
- **Service types** (Inpatient, Outpatient, Emergency, Surgery)
- **Prior authorization** tracking
- **Historical claim patterns**
- **Billing amounts** (log-normal distribution)
- **Rule-based denial logic** (~15-30% denial rate)

**Features**: 14 variables including temporal, financial, and clinical attributes

**Access the generator**: [Link to deployed tool or code]

## 🏗️ Architecture

```
┌─────────────────┐
│  Claim Data     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Baseline Model  │  (XGBoost/Neural Network)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MC Dropout      │  (T=100 forward passes)
│ Uncertainty     │
└────────┬────────┘
         │
         ▼
    ┌────┴────┐
    │ τ Check │ (Uncertainty Threshold)
    └────┬────┘
         │
    ┌────┴─────────┐
    │              │
    ▼              ▼
┌─────────┐  ┌──────────┐
│Auto     │  │ Human    │
│Decision │  │ Review   │
└─────────┘  └──────────┘
```

## 🚀 Methodology

### Step 1: Baseline Modeling
- Binary classification (Approved vs. Denied)
- Models: XGBoost, Feed-forward Neural Networks
- Feature engineering from RCM domain knowledge

### Step 2: Uncertainty Estimation (MC Dropout)
- Dropout layers active during inference
- Multiple stochastic forward passes (T=100)
- Calculate predictive mean and variance
- High variance → High uncertainty

### Step 3: ML with Rejection
```python
if uncertainty > threshold:
    defer_to_human_expert()
else:
    accept_model_prediction()
```

### Step 4: Evaluation
- **Predictive metrics**: Accuracy, Precision, Recall (on non-abstained cases)
- **Operational metrics**: Rejection rate, Coverage vs. Risk trade-off
- **Safety metrics**: Error analysis on deferred cases

## 📈 Expected Outcomes

- ✅ Calibrated uncertainty estimates for each prediction
- ✅ Reduced automation risk in safety-critical decisions
- ✅ Improved reliability compared to deterministic models
- ✅ Reusable framework for other healthcare AI applications

## 🛠️ Tech Stack

- **ML Frameworks**: PyTorch/TensorFlow, XGBoost, Scikit-learn
- **Uncertainty**: MC Dropout, Calibration metrics
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Deployment**: Streamlit/Gradio for demo interface

## 📁 Project Structure

```
├── data/
│   ├── synthetic_generator.py
│   └── generated_claims.csv
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_baseline_modeling.ipynb
│   └── 03_uncertainty_estimation.ipynb
├── src/
│   ├── models/
│   │   ├── baseline.py
│   │   └── mc_dropout.py
│   ├── utils/
│   │   ├── preprocessing.py
│   │   └── evaluation.py
│   └── deployment/
│       └── app.py
├── results/
│   ├── figures/
│   └── metrics/
├── requirements.txt
└── README.md
```

## 🎓 Academic Context

This project is part of **In25-S2-CS5801 - Advanced AI** coursework at the Department of Computer Science & Engineering, University of Moratuwa.

**Team RiseUp**:
- Aasir A.W.M. (258720U)
- Perera P.D.S. (258733L)
- Rizmy M.Z.M. (258736A)

## 📚 References

1. Guo et al. (2018) - "Predicting Health Insurance Claim Denials Using SVM and Logistic Regression"
2. Soni & Sharma (2021) - "A Deep Learning Framework using LSTMs for Time-Series Analysis of Medical Claims"
3. Chen, Li, & Zhang (2020) - "Interpretable ML for Payer-Specific Denial Pattern Identification"
4. Ravi & Krishnan (2022) - "Confidence Scoring of Medical Claim Adjudication using Softmax Probabilities"

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, fork the repository, and create pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or collaboration opportunities:
- GitHub Issues: [Link to issues page]
- Email: muhammedhu.25@cse.mrt.ac.lk

---

**Note**: This project uses synthetic data for research purposes and complies with HIPAA privacy standards. No real patient data is used or required.
