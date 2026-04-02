# 🩺 Diabetes Diagnosis Prediction (PyTorch)

This repository contains a Deep Learning approach to predicting diabetes using the **Pima Indians Diabetes Dataset**. The project evolved from an initial regression-style approach to a highly-tuned **Binary Classification model** optimized for **High-Sensitivity Medical Screening**.

---

## 📊 Final Results (Neural Network)

By tuning the model to a threshold of **0.29**, we achieved a *"Safety-First"* configuration that minimizes missed cases.

| Metric | Value |
|---|---|
| Accuracy | 73.91% |
| Recall (Sensitivity) | 95.00% |
| Precision | 58.00% |
| F1-Score | 0.72 |
| Negative Predictive Value | 95.9% |

### Confusion Matrix Breakdown

Tested on **115 validation samples**:

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | ✅ True Negatives: 47 | ❌ False Positives: 28 |
| **Actual Positive** | ❌ False Negatives: 2 | ✅ True Positives: 38 |

---

## 🛠️ Data Engineering Pipeline

The dataset underwent rigorous cleaning and feature selection:

1. **Imputation** — Replaced zeros in biological columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with median values.
2. **Feature Selection** — Dropped `Log_Pedigree`, `age_bin`, and `Insulin_to_glucose` to resolve multicollinearity (90%+ correlation) and reduce model noise.
3. **Scaling** — All features transformed using `StandardScaler` for zero mean and unit variance.
4. **Stratification** — Used stratified splitting **(70 / 15 / 15)** to maintain class balance across Train, Val, and Test sets.

---

## 🧠 Model Architecture

A custom PyTorch `nn.Module` optimized for tabular data:

| Layer | Units | Details |
|---|---|---|
| Layer 1 | 64 | Batch Normalization, ReLU, Dropout (0.3) |
| Layer 2 | 32 | Batch Normalization, ReLU, Dropout (0.2) |
| Layer 3 | 16 | ReLU |
| Output | 1 | Linear Logits |

### Training Strategy

| Component | Configuration |
|---|---|
| Loss | `BCEWithLogitsLoss` with `pos_weight=2.0` (prioritizes diabetic class) |
| Optimizer | Adam (`lr=0.005`) |
| Scheduler | `ReduceLROnPlateau` for automated learning rate decay |
| Early Stopping | Saves `best_model.pt` based on validation loss |

---

## 📈 ML Benchmarking

The Neural Network was compared against traditional ML models on the same scaled data:

| Model | Accuracy | Recall | F1-Score |
|---|---|---|---|
| **Neural Network (threshold=0.29)** | **73.9%** | **95.0%** | **0.72** |
| SVM | 77.9% | 77.7% | 0.71 |
| Random Forest | 77.2% | 77.7% | 0.70 |
| XGBoost | 75.3% | 59.2% | 0.62 |

> **Conclusion:** The Neural Network is the superior **clinical screening tool** — it ensures significantly fewer sick patients are missed compared to standard ML models.

---

## 📧 Contact

**Prashant Shukla**

- 📧 Email: [prashantshukla8851@gmail.com](mailto:prashantshukla8851@gmail.com)
- 💼 LinkedIn: [Prashant Shukla](https://www.linkedin.com/in/prashant-shukla)
- 🔗 GitHub: [@pr4sh4nt-shukla](https://github.com/pr4sh4nt-shukla)

---

⭐ *If this project helped you, please consider giving it a star!*
