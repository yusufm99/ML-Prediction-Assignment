# Census Income Classification 📊

## Project Overview
End‑to‑end machine learning pipeline using the 1994 UCI **Adult (Census Income)** dataset to predict whether an individual earns over \$50K/year. This assignment covers exploratory data analysis (EDA), data preprocessing, feature engineering, model training & evaluation, and fairness considerations.

## 🧾 Dataset
- **Source file**: `censusData.csv`
- **Origin**: UCI Machine Learning Repository (modified for educational purposes)

## 🔍 Problem Definition
- **Learning type**: Supervised  
- **Task**: Binary classification  
- **Target**: `income_binary` (0 = ≤50K, 1 = >50K)  

### Features:
- **Continuous**: `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`  
- **Categorical** (one-hot encoded): `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex_selfID`, `native-country`

---

## ⚙️ ML Workflow

### 1. Data Preprocessing
- Impute missing numeric values (e.g., age, hours-per-week) or drop rows for missing categorical entries  
- Encode the target variable using `LabelEncoder`  
- One-hot encode all categorical features  
- Standardize features using `StandardScaler`

### 2. Feature Selection
- Use a Random Forest classifier to obtain feature importances  
- Narrow to **top 20 features** to reduce noise and improve efficiency

### 3. Modeling
#### Logistic Regression (Baseline)
- Implement with `sklearn.linear_model.LogisticRegression`  
- Fast, interpretable, and achieved ~84% accuracy in original assignment (~84.2%)

#### Neural Network (TensorFlow/Keras)
- Architecture:
  - 3 hidden layers with ReLU activation
  - Batch normalization and dropout for regularization
- Optimizers: SGD and Adam
- Training: ~115 epochs, with metrics logging (e.g. every 5 epochs)
- Test accuracy close to ~83.9%

### 4. Evaluation
- Key metric: **accuracy**
- Analyze predicted probability distributions vs. actual outcomes
- Visualize training vs. validation loss and accuracy over epochs
- Compare model complexity vs. performance; highlight interpretability vs. accuracy trade-offs

### 5. Fairness & Bias Considerations
- Examine whether model outcomes correlate unfairly with demographic attributes  
- Highlight ethical implications in real‑world deployment

---

## 🛠 Tools & Technologies
- **Language**: Python 3  
- **Libraries**: pandas, NumPy, scikit‑learn, TensorFlow / Keras, Matplotlib, Seaborn
