# =============================================================================
# Titanic Survival Prediction using Logistic Regression
# Author: [Your Name]
# Dataset: https://www.kaggle.com/competitions/titanic/data
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)


# =============================================================================
# 1. Data Loading
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load the Titanic training dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


# =============================================================================
# 2. Data Preprocessing
# =============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values, drop irrelevant columns,
    and encode categorical features.
    """
    # Fill missing Age with column mean
    df['Age'].fillna(df['Age'].mean(), inplace=True)

    # Fill missing Embarked with mode
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Drop Cabin (too many missing values — >77%), Ticket (high cardinality)
    df = df.drop(columns=['Cabin', 'Ticket'], axis=1)

    # Encode Sex: female=0, male=1
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})

    # Encode Embarked: S=0, Q=1, C=2
    df['Embarked'] = df['Embarked'].map({'S': 0, 'Q': 1, 'C': 2})

    print(f"\nAfter preprocessing — missing values:\n{df.isnull().sum()}")
    return df


# =============================================================================
# 3. Exploratory Data Analysis
# =============================================================================

def plot_eda(df: pd.DataFrame) -> None:
    """Visualise survival rates by key features."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Titanic Dataset — Exploratory Data Analysis", fontsize=16)

    # Overall survival
    sns.countplot(x='Survived', data=df, ax=axes[0, 0],
                  palette=['salmon', 'steelblue'])
    axes[0, 0].set_title("Overall Survival")
    axes[0, 0].set_xticklabels(['Not Survived (0)', 'Survived (1)'])

    # Survival by Sex
    sns.countplot(x='Sex', hue='Survived', data=df, ax=axes[0, 1],
                  palette=['salmon', 'steelblue'])
    axes[0, 1].set_title("Survival by Gender")
    axes[0, 1].set_xticklabels(['Female (0)', 'Male (1)'])
    axes[0, 1].legend(title='Survived', labels=['No', 'Yes'])

    # Survival by Pclass
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[1, 0],
                  palette=['salmon', 'steelblue'])
    axes[1, 0].set_title("Survival by Passenger Class")
    axes[1, 0].legend(title='Survived', labels=['No', 'Yes'])

    # Age distribution by survival
    df[df['Survived'] == 0]['Age'].plot(
        kind='hist', bins=30, alpha=0.6, color='salmon',
        label='Not Survived', ax=axes[1, 1]
    )
    df[df['Survived'] == 1]['Age'].plot(
        kind='hist', bins=30, alpha=0.6, color='steelblue',
        label='Survived', ax=axes[1, 1]
    )
    axes[1, 1].set_title("Age Distribution by Survival")
    axes[1, 1].set_xlabel("Age")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=150)
    plt.show()
    print("EDA plots saved as 'eda_plots.png'")

    # Fare distribution by survival
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Survived', y='Fare', data=df,
                palette=['salmon', 'steelblue'])
    plt.title("Fare Distribution by Survival")
    plt.xticks([0, 1], ['Not Survived', 'Survived'])
    plt.tight_layout()
    plt.savefig("fare_by_survival.png", dpi=150)
    plt.show()
    print("Fare plot saved as 'fare_by_survival.png'")

    # Correlation heatmap
    plt.figure(figsize=(9, 7))
    numeric_df = df.select_dtypes(include=np.number)
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f',
                cmap='Blues', square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=150)
    plt.show()
    print("Heatmap saved as 'correlation_heatmap.png'")


# =============================================================================
# 4. Feature / Target Split
# =============================================================================

def split_features_target(df: pd.DataFrame):
    """Drop identifiers and separate features from the target."""
    X = df.drop(columns=['Survived', 'Name', 'PassengerId'], axis=1)
    Y = df['Survived']
    print(f"\nFeatures: {X.shape} | Target: {Y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    return X, Y


# =============================================================================
# 5. Train / Test Split
# =============================================================================

def split_data(X, Y, test_size=0.2, random_state=2):
    """Split data into training and test sets."""
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, Y_train, Y_test


# =============================================================================
# 6. Model Training
# =============================================================================

def train_model(X_train, Y_train):
    """Train a Logistic Regression classifier."""
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    print("Model training complete.")
    return model


# =============================================================================
# 7. Model Evaluation
# =============================================================================

def evaluate_model(model, X_train, Y_train, X_test, Y_test) -> None:
    """Accuracy, classification report, and confusion matrix."""
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)

    print(f"\nTraining Accuracy : {accuracy_score(Y_train, train_preds):.4f}")
    print(f"Test     Accuracy : {accuracy_score(Y_test,  test_preds):.4f}")

    print("\nClassification Report (Test Set):")
    print(classification_report(
        Y_test, test_preds,
        target_names=['Not Survived', 'Survived']
    ))

    # Confusion matrix
    cm = confusion_matrix(Y_test, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix saved as 'confusion_matrix.png'")

    # Feature coefficients (logistic regression interpretability)
    coef_df = pd.Series(
        model.coef_[0],
        index=X_train.columns
    ).sort_values()
    plt.figure(figsize=(7, 5))
    coef_df.plot(kind='barh', color=['salmon' if c < 0 else 'steelblue'
                                     for c in coef_df])
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title("Feature Coefficients — Logistic Regression")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.savefig("feature_coefficients.png", dpi=150)
    plt.show()
    print("Feature coefficients saved as 'feature_coefficients.png'")


# =============================================================================
# 8. Predictive System
# =============================================================================

def predict_survival(model, input_data: tuple) -> str:
    """
    Predict whether a passenger would survive the Titanic disaster.

    Parameters
    ----------
    input_data : tuple of 7 values:
        (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)

        Pclass   : 1 = First, 2 = Second, 3 = Third class
        Sex      : female=0, male=1
        Age      : age in years
        SibSp    : number of siblings / spouses aboard
        Parch    : number of parents / children aboard
        Fare     : ticket fare paid
        Embarked : S=0, Q=1, C=2
    """
    arr = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(arr)

    if prediction[0] == 0:
        return "💀 The passenger would NOT survive."
    else:
        return "🛟 The passenger would SURVIVE."


# =============================================================================
# Main Pipeline
# =============================================================================

if __name__ == "__main__":
    DATA_PATH = "titanic_train.csv"   # update path if needed

    # 1. Load
    df = load_data(DATA_PATH)
    print("\nFirst 5 rows:\n", df.head())

    # 2. Preprocess
    df = preprocess_data(df)

    # 3. EDA
    plot_eda(df)

    # 4. Features & Target
    X, Y = split_features_target(df)

    # 5. Split
    X_train, X_test, Y_train, Y_test = split_data(X, Y)

    # 6. Train
    model = train_model(X_train, Y_train)

    # 7. Evaluate
    evaluate_model(model, X_train, Y_train, X_test, Y_test)

    # 8. Predict a sample passenger
    # Jack: 3rd class, male, age 22, 1 sibling, 0 parents, fare 7.25, embarked S
    sample = (3, 1, 22.0, 1, 0, 7.25, 0)
    result = predict_survival(model, sample)
    print(f"\nSample Prediction (Jack — 3rd class, male, 22):\n{result}")

    # Rose: 1st class, female, age 17, 1 sibling, 2 parents, fare 100.0, embarked C
    sample_rose = (1, 0, 17.0, 1, 2, 100.0, 2)
    result_rose = predict_survival(model, sample_rose)
    print(f"\nSample Prediction (Rose — 1st class, female, 17):\n{result_rose}")
