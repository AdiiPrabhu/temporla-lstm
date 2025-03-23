import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load test data (clearly defined)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

print("Test data loaded successfully.")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Load trained model (clearly defined)
model = tf.keras.models.load_model('best_tuned_lstm.h5')

print("Model loaded successfully.")

# Generate predictions clearly
y_pred_probs = model.predict(X_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print(f"y_pred_probs shape: {y_pred_probs.shape}")
print(f"y_pred shape: {y_pred.shape}")

# Verify array shapes clearly
print(f"y_test unique values: {np.unique(y_test)}")
print(f"y_pred unique values: {np.unique(y_pred)}")

# Compute evaluation metrics clearly
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_probs)

# Clearly print evaluation metrics
print("\nModel Evaluation Metrics clearly defined:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# ROC Curve (clearly defined visualization)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2, label=f'LSTM (ROC-AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Depression Detection)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()

# Save ROC curve clearly
plt.savefig('roc_curve_lstm.png', dpi=300)
plt.show()
