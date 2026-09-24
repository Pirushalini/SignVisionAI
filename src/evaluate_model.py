import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# Load dataset
df = pd.read_csv("data/asl_hand_landmarks.csv")

X = df.drop(columns=["label"])
y = df["label"]


# Same split used in previous experiments
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# MLP model
model = Pipeline([
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


# Train
print("Training MLP...")
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Classification report
print("\nClassification Report")
print("=" * 70)
print(classification_report(y_test, y_pred))


# Confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(14, 14))

disp.plot(
    ax=ax,
    xticks_rotation="vertical",
    values_format="d"
)

ax.set_title("MLP Confusion Matrix - ASL Sign Recognition")

plt.tight_layout()

plt.savefig(
    "results/mlp_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved to:")
print("results/mlp_confusion_matrix.png")