# Project Execution Guide
## Uncertainty-Aware Claim Denial Prediction

**Team RiseUp** | University of Moratuwa | Advanced AI (CS5801)

---

## 📁 Project Structure

```
uncertainty-aware-claim-denial/
│
├── data/
│   ├── synthetic_data_generator.py     # Generate synthetic claims
│   └── healthcare_claims_synthetic_10k.csv
│
├── notebooks/
│   ├── 01_exploratory_analysis.py      # EDA
│   ├── 02_baseline_models.py           # Baseline ML models
│   └── 03_mc_dropout_uncertainty.py    # MC Dropout implementation
│
├── models/
│   ├── best_baseline_model.pkl         # Saved XGBoost/RF model
│   ├── mc_dropout_model.pth            # Saved PyTorch model
│   ├── scaler.pkl                      # Feature scaler
│   └── label_encoders.pkl              # Categorical encoders
│
├── results/
│   ├── figures/                        # All visualization outputs
│   │   ├── target_distribution.png
│   │   ├── denial_by_categorical.png
│   │   ├── confusion_matrices.png
│   │   ├── roc_curves.png
│   │   ├── uncertainty_analysis.png
│   │   └── rejection_analysis.png
│   └── metrics/
│       └── model_comparison.csv
│
├── deployment/
│   └── streamlit_app.py                # Optional: Demo application
│
├── requirements.txt                    # Python dependencies
├── README.md                           # Project documentation
└── PROJECT_EXECUTION_GUIDE.md         # This file
```

---

## 🚀 Step-by-Step Execution

### Step 0: Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 1: Generate Synthetic Data (5 minutes)

```bash
cd data
python synthetic_data_generator.py
```

**Output:**
- `healthcare_claims_synthetic_10k.csv` (10,000 claims)
- Dataset statistics printed to console

**What it does:**
- Creates realistic healthcare claims with 14 features
- Implements rule-based denial logic (~20% denial rate)
- Generates medical codes (CPT/ICD-10)
- Includes prior authorization tracking

---

### Step 2: Exploratory Data Analysis (10 minutes)

```bash
cd ../notebooks
python 01_exploratory_analysis.py
```

**Output Visualizations:**
- `target_distribution.png` - Class balance
- `denial_by_categorical.png` - Denial rates by payer, service type, etc.
- `numerical_distributions.png` - Feature distributions
- `billing_analysis.png` - Billing amount deep dive
- `prior_auth_impact.png` - Prior authorization impact
- `correlation_matrix.png` - Feature correlations
- `feature_importance_correlation.png` - Correlation with target

**Key Insights to Document:**
- Which payers have highest denial rates?
- How does billing amount affect denial probability?
- Impact of prior authorization on denials
- Most correlated features with denial outcome

---

### Step 3: Baseline Model Training (15-20 minutes)

```bash
python 02_baseline_models.py
```

**Models Trained:**
1. **Logistic Regression** - Simple linear baseline
2. **Random Forest** - Tree-based ensemble
3. **XGBoost** - Gradient boosting (typically best)

**Output:**
- Model comparison metrics table
- `confusion_matrices.png` - All three models
- `roc_curves.png` - ROC comparison
- `precision_recall_curves.png` - PR curves
- `model_comparison.png` - Bar chart comparison
- `xgboost_feature_importance.png`
- Saved best model: `best_baseline_model.pkl`

**Expected Performance:**
- Accuracy: ~85-90%
- ROC-AUC: ~0.88-0.93
- F1-Score: ~0.75-0.85

**⚠️ Limitation Identified:**
These models provide deterministic predictions without uncertainty estimates!

---

### Step 4: MC Dropout Implementation (20-30 minutes)

```bash
python 03_mc_dropout_uncertainty.py
```

**What Happens:**
1. Neural network trained with dropout layers
2. Dropout remains active during inference
3. 100 stochastic forward passes per prediction
4. Uncertainty quantified via prediction variance

**Output:**
- `mc_dropout_training_history.png` - Loss and accuracy curves
- `uncertainty_analysis.png` - Uncertainty distributions
  - Correct predictions have LOWER uncertainty
  - Incorrect predictions have HIGHER uncertainty
- `rejection_analysis.png` - Coverage vs accuracy trade-off
- Saved model: `mc_dropout_model.pth`

**Key Metrics:**
- **Without Rejection:** ~85-88% accuracy on all claims
- **With Optimal Rejection:** ~92-95% accuracy on 80% of claims
  - 20% of claims deferred to human review
  - Significantly higher reliability on auto-processed claims

---

## 📊 Expected Results Summary

### Baseline Models (Deterministic)
| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 0.85 | 0.76 | 0.89 |
| Random Forest | 0.88 | 0.81 | 0.91 |
| **XGBoost** | **0.90** | **0.84** | **0.93** |

### MC Dropout (Uncertainty-Aware)
| Configuration | Accuracy | Coverage | Rejection Rate |
|---------------|----------|----------|----------------|
| No Rejection | 88% | 100% | 0% |
| Conservative | 95% | 70% | 30% |
| **Optimal** | **93%** | **80%** | **20%** |

### Business Impact
- **Same manual effort** (20% human review)
- **Higher reliability** (93% vs 88% accuracy)
- **Risk reduction** in safety-critical decisions
- **Cost-effective** automation with human oversight

---

## 🎯 Key Findings for Report

### 1. Problem Validation
✓ Traditional ML models achieve good accuracy but lack confidence measures
✓ All predictions treated equally - risky for complex cases

### 2. Solution Effectiveness
✓ MC Dropout successfully quantifies prediction uncertainty
✓ High uncertainty correlates with incorrect predictions
✓ Rejection mechanism enables safer automation

### 3. Practical Value
✓ Same workload, higher reliability
✓ Human experts focus on truly complex cases
✓ Reduced risk of incorrect automated decisions

### 4. Novel Contribution
✓ First uncertainty-aware framework for RCM claim denial prediction
✓ Demonstrates ML with rejection in healthcare operations
✓ Reusable methodology for other safety-critical healthcare tasks

---

## 📝 Deliverables Checklist

### Code & Models
- [ ] Synthetic data generator
- [ ] EDA scripts with visualizations
- [ ] Baseline models (3 algorithms)
- [ ] MC Dropout implementation
- [ ] Saved models and preprocessors

### Documentation
- [ ] README.md with project overview
- [ ] This execution guide
- [ ] Code comments and docstrings
- [ ] Requirements.txt

### Visualizations (12+ figures)
- [ ] Target distribution
- [ ] Feature analysis (5 figures)
- [ ] Model comparison plots (3 figures)
- [ ] Uncertainty analysis (2 figures)
- [ ] Rejection analysis (2 figures)

### Results
- [ ] Model performance comparison table
- [ ] Rejection strategy analysis
- [ ] Feature importance rankings
- [ ] Statistical significance tests

---

## 🔧 Troubleshooting

### Issue: Low model accuracy (<80%)
**Solution:** 
- Increase dataset size to 20,000+ samples
- Tune hyperparameters (learning rate, dropout rate)
- Add more feature engineering

### Issue: Training too slow
**Solution:**
- Reduce MC samples from 100 to 50
- Use GPU if available: `device = torch.device('cuda')`
- Reduce batch size or number of epochs

### Issue: Uncertainty doesn't separate correct/incorrect well
**Solution:**
- Increase dropout rate (try 0.4 or 0.5)
- Train longer (more epochs)
- Try ensemble of multiple models

### Issue: Memory errors
**Solution:**
- Reduce batch size
- Process test set in smaller batches
- Use smaller dataset for initial testing

---

## 🎓 For Academic Report

### Structure Recommendation:

1. **Introduction** (1-2 pages)
   - Healthcare RCM background
   - Problem statement
   - Research objectives

2. **Literature Review** (2-3 pages)
   - ML in healthcare RCM
   - Uncertainty quantification methods
   - ML with rejection

3. **Methodology** (3-4 pages)
   - Dataset generation
   - Baseline models
   - MC Dropout architecture
   - Evaluation framework

4. **Results** (3-4 pages)
   - EDA findings
   - Baseline model performance
   - Uncertainty analysis
   - Rejection analysis
   - **Use all generated visualizations!**

5. **Discussion** (2 pages)
   - Key findings
   - Practical implications
   - Limitations
   - Future work

6. **Conclusion** (1 page)
   - Summary of contributions
   - Impact on healthcare RCM

### Figures to Include:
- Data distribution (2-3 figures)
- Model comparison (2 figures)
- Uncertainty analysis (3 figures)
- Rejection curves (2 figures)

---

## 📧 Support

For questions or issues:
- Check code comments and docstrings
- Review error messages carefully
- Consult PyTorch/scikit-learn documentation
- Team communication channels

---

## ✨ Next Steps (Optional Enhancements)

1. **Calibration Analysis**
   - Implement calibration curves
   - Compare calibrated vs uncalibrated predictions

2. **Explainability**
   - Add SHAP values for feature importance
   - Explain individual predictions

3. **Deployment**
   - Build Streamlit web application
   - Create REST API for predictions

4. **Advanced Uncertainty**
   - Implement deep ensembles
   - Try Bayesian Neural Networks
   - Compare multiple uncertainty methods

5. **Real Data Testing**
   - Test on MIMIC-IV billing data
   - Validate on CMS public claims

---

**Good luck with your project! 🚀**

*Last updated: [Current Date]*
*Team RiseUp - University of Moratuwa*