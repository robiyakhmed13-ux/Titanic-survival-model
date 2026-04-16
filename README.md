# 🚢 Titanic Survival Prediction

A machine learning project that predicts whether a **Titanic passenger would survive or not**, using **Logistic Regression** based on personal attributes such as age, gender, passenger class, and fare.

---

## 📌 Project Overview

The sinking of the Titanic in 1912 is one of history's most famous maritime disasters. This project uses the classic Titanic dataset to train a binary classification model — exploring the famous pattern that survival was heavily influenced by factors like gender ("women and children first") and passenger class.

| Item | Detail |
|------|--------|
| **Algorithm** | Logistic Regression |
| **Task** | Binary Classification |
| **Dataset** | [Titanic – Kaggle Competition](https://www.kaggle.com/competitions/titanic/data) |
| **Target** | `Survived` — Not Survived (0) / Survived (1) |

---

## 📂 Project Structure

```
titanic_survival_prediction/
│
├── titanic_survival_prediction.ipynb   # Jupyter Notebook (full walkthrough)
├── titanic_survival_prediction.py      # Clean Python script
├── requirements.txt                    # Dependencies
├── titanic_train.csv                   # Dataset (download from Kaggle)
├── eda_plots.png                       # EDA — survival by gender, class, age
├── fare_by_survival.png                # Fare distribution by survival
├── correlation_heatmap.png             # Feature correlation heatmap
├── confusion_matrix.png                # Confusion matrix
├── feature_coefficients.png            # Logistic Regression feature weights
└── README.md
```

---

## 📊 Dataset Features

| Feature | Description |
|---------|-------------|
| `PassengerId` | Unique passenger ID (dropped — identifier only) |
| `Survived` | ✅ **Target** — 0 = Not Survived, 1 = Survived |
| `Pclass` | Ticket class — 1st, 2nd, 3rd |
| `Name` | Passenger name (dropped — high cardinality) |
| `Sex` | Gender — female=0, male=1 |
| `Age` | Age in years (missing values filled with mean) |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Ticket` | Ticket number (dropped — high cardinality) |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number (dropped — 77%+ missing) |
| `Embarked` | Port of embarkation — S=0, Q=1, C=2 |

---

## 🧹 Data Preprocessing

| Issue | Solution |
|-------|----------|
| `Age` — ~20% missing | Filled with **column mean** |
| `Embarked` — 2 missing | Filled with **mode** (most frequent port) |
| `Cabin` — 77% missing | **Dropped** — too sparse to be useful |
| `Ticket` — high cardinality | **Dropped** — alphanumeric, no predictive structure |
| `Name`, `PassengerId` | **Dropped** — identifiers, not features |

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `titanic_train.csv` from [Kaggle](https://www.kaggle.com/competitions/titanic/data) (filename: `train.csv`, rename to `titanic_train.csv`) and place it in the project root.

### 4. Run
```bash
python titanic_survival_prediction.py
```

---

## 🔄 Pipeline

```
Raw CSV Data (891 passengers)
    │
    ▼
Handle Missing Values (Age → mean, Embarked → mode)
    │
    ▼
Drop Irrelevant Columns (Cabin, Ticket, Name, PassengerId)
    │
    ▼
Encode Categoricals (Sex, Embarked)
    │
    ▼
EDA — Survival by gender, class, age, fare
    │
    ▼
Feature / Target Split
    │
    ▼
Train / Test Split (80% / 20%)
    │
    ▼
Logistic Regression Training
    │
    ▼
Accuracy + Classification Report + Confusion Matrix
    │
    ▼
Feature Coefficient Plot (model interpretability)
    │
    ▼
Single-passenger Survival Prediction
```

---

## 📈 Results

| Split | Accuracy |
|-------|----------|
| Training | ~80% |
| Test | ~79% |

---

## 🔑 Key Findings (EDA)

- **Gender** is the strongest predictor — female passengers had a much higher survival rate ("women and children first")
- **Passenger Class** matters — 1st class passengers survived at a significantly higher rate than 3rd class
- **Fare** correlates with survival — higher fare (proxy for wealth/class) meant better odds
- **Age** shows that children had a slightly higher survival rate

---

## 🔮 Sample Predictions

```python
# Jack — 3rd class, male, age 22, fare 7.25, embarked at Southampton
sample_jack = (3, 1, 22.0, 1, 0, 7.25, 0)
# Output: 💀 The passenger would NOT survive.

# Rose — 1st class, female, age 17, fare 100.0, embarked at Cherbourg
sample_rose = (1, 0, 17.0, 1, 2, 100.0, 2)
# Output: 🛟 The passenger would SURVIVE.
```

---

## 🛠️ Tech Stack

- **Python 3.x**
- **pandas / numpy** — data processing
- **scikit-learn** — Logistic Regression, train/test split, metrics
- **seaborn / matplotlib** — visualization

---

## 🚀 Future Improvements

- [ ] Engineer new features: title from `Name` (Mr, Mrs, Miss, Master), family size (`SibSp + Parch + 1`), `isAlone` flag
- [ ] Try Random Forest or XGBoost for improved accuracy
- [ ] Hyperparameter tuning with `GridSearchCV`
- [ ] Cross-validation (k-fold) for more reliable evaluation
- [ ] Submit predictions on the Kaggle test set

---

## 📄 License

MIT License

---

## 🙋 Author

**Akhmedova Robiyakhon**  
[GitHub](https://github.com/robiyakhmed13-ux)) | [LinkedIn] (www.linkedin.com/in/robiyakhon-akhmedova-06467b324)
