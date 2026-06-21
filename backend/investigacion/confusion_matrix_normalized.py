import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import joblib
import tensorflow as tf

val_file = "val.csv"
feature_extractor = "Virchow2"
model_classifier = "Red Propia"
features_val = "./features/val/features_virchow2val_norm.npy"

df_val = pd.read_csv(val_file)
X_val = np.load(features_val)
y_val = df_val['label'].values

scaler_path = f"./trained_models/scaler_{feature_extractor}_norm.joblib"
scaler = joblib.load(scaler_path)
X_val_s = scaler.transform(X_val)

model_path = f"./trained_models/{model_classifier}_{feature_extractor}_norm.keras"
model = tf.keras.models.load_model(model_path)

y_probability= model.predict(X_val_s, verbose=0)
predictions = np.argmax(y_probability, axis=1)

conf_matrix = confusion_matrix(y_val, predictions)

plt.figure(figsize=(10,8))
classes = ['NC', 'Gleason 3', 'Gleason 4', 'Gleason 5']
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=False)
plt.title(f"Confusion matrix {feature_extractor} & {model_classifier} normalizada", pad=10)
plt.ylabel('Ground Truth')
plt.xlabel('Model prediction')
output=f"confusion_matrix_{feature_extractor}_{model_classifier}_SICAPv2_norm.png"
plt.savefig(output, dpi=300)