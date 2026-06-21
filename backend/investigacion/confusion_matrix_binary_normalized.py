import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import tensorflow as tf

def binary(labels):
    return (np.array(labels) > 0).astype(int)

val_file = "val.csv"
feature_extractor = "Virchow2"
model_classifier = "Red Propia"
features_val = "./features/val/features_virchow2val_norm.npy"

df_val = pd.read_csv(val_file)
X_val = np.load(features_val)
y_val = df_val['label'].values
y_val_binary = binary(y_val)

scaler_path = f"./trained_models/scaler_{feature_extractor}_norm.joblib"
scaler = joblib.load(scaler_path)
X_val_s = scaler.transform(X_val)

if model_classifier == "Red Propia":
    model_path = f"./trained_models/{model_classifier}_{feature_extractor}_norm.keras"
    model = tf.keras.models.load_model(model_path)
    y_probability= model.predict(X_val_s, verbose=0)
else:
    model_path = f"./trained_models/{model_classifier}_{feature_extractor}_norm.joblib"
    model = joblib.load(model_path)
    y_probability= model.predict_proba(X_val_s)

y_probability_binary= y_probability[:, 1:].sum(axis=1)
predictions_binary = (y_probability_binary > 0.5).astype(int)

conf_matrix = confusion_matrix(y_val_binary, predictions_binary)

plt.figure(figsize=(8,6))
classes = ['Sano (NC)', 'Cáncer']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
plt.title(f"Matriz de confusión binaria {feature_extractor} & {model_classifier} normalizada", pad=10)
plt.ylabel('Ground Truth')
plt.xlabel('Model prediction')
output=f"binary_confusion_matrix_{feature_extractor}_{model_classifier}_SICAPv2_norm.png"
plt.savefig(output, dpi=300)