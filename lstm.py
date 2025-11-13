# =====================================================
# Credit Card Fraud Detection using LSTM (TensorFlow/Keras)
# =====================================================

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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# -----------------------------------------------------
# 1. Load dataset
# -----------------------------------------------------
data = pd.read_csv("creditcard_final_cleaned.csv")

# Define target and features
target = "Class" if "Class" in data.columns else data.columns[-1]
features = data.drop(columns=[target])
labels = data[target]

# Handle categorical features if any
features = pd.get_dummies(features, drop_first=True)


# -----------------------------------------------------
# 2. Train-test split + feature scaling
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------------------------------
# 3. Reshape data for LSTM (samples, timesteps, features)
# -----------------------------------------------------
X_train_seq = np.expand_dims(X_train_scaled, axis=1)
X_test_seq = np.expand_dims(X_test_scaled, axis=1)


# -----------------------------------------------------
# 4. Build LSTM model
# -----------------------------------------------------
lstm_model = Sequential([
    LSTM(64, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")   # Binary classification output
])

lstm_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# -----------------------------------------------------
# 5. Train model
# -----------------------------------------------------
history = lstm_model.fit(
    X_train_seq, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.2,
    verbose=1
)


# -----------------------------------------------------
# 6. Plot training history
# -----------------------------------------------------
plt.figure(figsize=(12,5))

# Loss curve
plt.subplot(1,2,1)
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy curve
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()


# -----------------------------------------------------
# 7. Evaluate model
# -----------------------------------------------------
y_scores = lstm_model.predict(X_test_seq).ravel()
y_pred = (y_scores >= 0.5).astype(int)

print("Test Accuracy :", accuracy_score(y_test, y_pred))
print("Test Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Test Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("Test F1-score :", f1_score(y_test, y_pred, zero_division=0))
print("\nDetailed Report:\n", classification_report(y_test, y_pred))
