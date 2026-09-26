from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Files are saved beside this script.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "Telco-Customer-Churn.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)


def load_data():
    """Download the sample CSV if it isn't already in this folder."""
    if not DATA_PATH.exists():
        print("Downloading the IBM Telco churn dataset...")
        urlretrieve(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric; blank values become missing values.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].replace(" ", pd.NA),
            errors="coerce",
        )

    return df


def explore_data(df):
    """Print summary statistics and save churn charts."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Customers: {len(df):,}")
    print(f"Overall churn rate: {churn_rate:.1%}")

    print("\nChurn rate by contract:")
    contract_rates = (
        df.groupby("Contract")["Churn"]
        .apply(lambda values: (values == "Yes").mean())
        .sort_values(ascending=False)
    )
    print((contract_rates * 100).round(1).astype(str) + "%")

    # Compare churn across groups of customer tenure.
    tenure_groups = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0–12 months", "13–24 months", "25–48 months", "49–72 months"],
    )
    print("\nChurn rate by tenure group:")
    tenure_rates = (
        df.assign(tenure_group=tenure_groups)
        .groupby("tenure_group", observed=False)["Churn"]
        .apply(lambda values: (values == "Yes").mean())
    )
    print((tenure_rates * 100).round(1).astype(str) + "%")

    # Chart 1: number of customers who stayed or churned.
    counts = df["Churn"].value_counts().reindex(["No", "Yes"], fill_value=0)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False)
    plt.title("Customer churn counts")
    plt.xlabel("Churn")
    plt.ylabel("Number of customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_counts.png", dpi=150)
    plt.close()

    # Chart 2: churn rate by contract type.
    plt.figure(figsize=(7, 4))
    sns.barplot(data=df, x="Contract", y=df["Churn"].eq("Yes").astype(int))
    plt.title("Churn rate by contract type")
    plt.xlabel("Contract")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
    plt.close()

    print(f"\nCharts saved in: {OUTPUT_DIR}")


def train_model(df):
    """Train a basic logistic regression churn model."""
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    y = df["Churn"].map({"No": 0, "Yes": 1})

    numeric_columns = X.select_dtypes(include="number").columns
    categorical_columns = X.select_dtypes(exclude="number").columns

    numeric_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_steps = Pipeline([
        ("fill_missing", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_steps, numeric_columns),
        ("categorical", categorical_steps, categorical_columns),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )

    print("\nBaseline logistic regression results")
    print(f"ROC AUC:   {auc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 score:  {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Churned"],
        zero_division=0,
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Stayed", "Churned"],
        cmap="Blues",
    )
    plt.title("Churn model confusion matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    df = load_data()
    explore_data(df)
    train_model(df)


if __name__ == "__main__":
    main()
