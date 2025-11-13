# -------------------------------
# Credit Card Fraud Detection using MLP
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam


# -------------------------------
# 1. Load and prepare dataset
# -------------------------------
data = pd.read_csv("creditcard_final_cleaned.csv")

# Identify target variable (fraud / non-fraud)
label_col = "Class" if "Class" in data.columns else data.columns[-1]
X = data.drop(columns=[label_col])
y = data[label_col]

# Encode categorical features if present
X = pd.get_dummies(X, drop_first=True)

# Split into training and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -------------------------------
# 2. Build Neural Network Model
# -------------------------------
mlp_model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")   # binary output
])

mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -------------------------------
# 3. Train the Model
# -------------------------------
history = mlp_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)


# -------------------------------
# 4. Visualize Training Progress
# -------------------------------
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Trend")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy Trend")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# -------------------------------
# 5. Model Evaluation
# -------------------------------
y_prob = mlp_model.predict(X_test).ravel()
y_pred = (y_prob >= 0.5).astype(int)

print("Test Accuracy :", accuracy_score(y_test, y_pred))
print("Test Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Test Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("Test F1-score :", f1_score(y_test, y_pred, zero_division=0))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))
