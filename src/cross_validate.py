import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# Load dataset
df = pd.read_csv("data/asl_hand_landmarks.csv")

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset shape:", df.shape)
print("Classes:", y.nunique())


# 5-Fold Stratified Cross Validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            random_state=42
        ))
    ]),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=300,
            early_stopping=True,
            random_state=42
        ))
    ])
}


results = []


for name, model in models.items():

    print(f"\nRunning {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        n_jobs=-1
    )

    accuracy_mean = scores["test_accuracy"].mean()
    accuracy_std = scores["test_accuracy"].std()

    f1_mean = scores["test_f1_macro"].mean()
    f1_std = scores["test_f1_macro"].std()

    results.append({
        "Model": name,
        "Mean Accuracy": accuracy_mean,
        "Accuracy Std": accuracy_std,
        "Mean Macro F1": f1_mean,
        "Macro F1 Std": f1_std
    })

    print(f"Mean Accuracy : {accuracy_mean:.4f}")
    print(f"Accuracy Std  : {accuracy_std:.4f}")
    print(f"Mean Macro F1 : {f1_mean:.4f}")
    print(f"Macro F1 Std  : {f1_std:.4f}")


# Results table
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 70)

print(results_df.to_string(index=False))


# Save results
results_df.to_csv(
    "results/cross_validation_results.csv",
    index=False
)

print("\nResults saved to:")
print("results/cross_validation_results.csv")
