import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


# Load dataset
df = pd.read_csv("data/asl_hand_landmarks.csv")

X = df.drop(columns=["label"])
y = df["label"]

print("Dataset shape:", df.shape)
print("Number of classes:", y.nunique())


# Final MLP pipeline
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


# Train on complete dataset
print("\nTraining final MLP model...")

model.fit(X, y)


# Save complete pipeline
model_path = "models/sign_language_mlp.pkl"

joblib.dump(model, model_path)


print("\nFinal model trained successfully.")
print("Model saved to:")
print(model_path)