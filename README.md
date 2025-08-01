# Income Classification 📊

End-to-end machine-learning pipeline.  
Using the **1994 UCI Adult (“Census Income”)** data, we predict whether an individual earns **> \$50 K / year** and compare four classical & ensemble algorithms.

---

## 🗂 Dataset

| Item | Value |
|------|-------|
| **File** | `data/censusData.csv` |
| **Source** | [UCI Machine-Learning Repository ▸ Adult](https://archive.ics.uci.edu/ml/datasets/adult) <br>(slightly cleaned for education) |
| **Size** | 32 561 rows × 15 raw features |

---

## 🔍 Problem Definition

| Aspect | Details |
|--------|---------|
| **Learning type** | Supervised |
| **Task** | Binary classification |
| **Target** | `income_binary` → 0 = ≤ \$50 K, 1 = > \$50 K |
| **Feature groups** | • **Continuous (6)** `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`  <br>• **Categorical (9)** `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`, `income` |

---

## ⚙️ ML Workflow

1. **Data Pre-processing**
   - Drop rows with placeholder “?” entries.  
   - Label-encode **target**; one-hot encode all categorical predictors.  
   - **Standardize** continuous variables with `StandardScaler`.

2. **Feature Selection**  
   - Use SelectKBest with mutual-information scores to keep the top 20 most informative features.  

3. **Model Training**
   | Model | Library / API | Key Hyper-params |
   |-------|--------------|------------------|
   | Logistic Regression | `sklearn.linear_model.LogisticRegression` | `solver='lbfgs', max_iter=1000` |
   | Random Forest | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=300, max_depth=None` |
   | Gradient Boosting (GBDT) | `sklearn.ensemble.GradientBoostingClassifier` | `n_estimators=400, learning_rate=0.05` |
   | Stacking Ensemble | `sklearn.ensemble.StackingClassifier` | Base: LR + RF + GBDT → Meta: LR |

4. **Evaluation**
   - **Confusion matrices**, **accuracy**, **precision-recall** curves, and **ROC-AUC**.
   - Training-vs-validation learning curves to spot over/under-fitting.
   - Basic **fairness probe**: check error rates across `sex` & `race`.

---

## 📈 Results

| Model | Accuracy (%) | Notes |
|-------|-------------|-------|
| Logistic Regression | **84.5** | Fast & highly interpretable |
| Random Forest | **86.3** | Captures non-linear splits; small memory cost |
| Gradient Boosting | **87.3** | Best single model; slower to train |
| Stacking Ensemble | **87.4** | Marginal lift above GBDT |

*(Accuracies computed on held-out test set of 6 513 records.)*

---

## 🔬 Fairness & Bias Checks

*Preliminary* disparate-impact analysis shows higher false-negative rates for women and certain racial groups. Mitigation strategies (re-sampling, re-weighting, post-processing thresholds) are discussed in `notebooks/04_fairness_analysis.ipynb`.

---

## 🛠 Tech Stack

- **Python 3.11**
- **pandas**, **NumPy**
- **scikit-learn**
- **Matplotlib**, **Seaborn**



