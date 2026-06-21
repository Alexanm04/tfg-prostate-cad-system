import pandas as pd
import numpy as np
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score)
import joblib
import tensorflow as tf

def binary(labels):
    return (np.array(labels) > 0).astype(int)

df_val = pd.read_csv('val.csv')
y_val = binary(df_val['label'].values)

features = {
 "ResNet-50": ("./features/val/features_resnet50val_norm.npy"),
 "UNI2-h": ("./features/val/features_uni2val_norm.npy"),
 "Prov-GigaPath": ("./features/val/features_gigapathval_norm.npy"),
 "Virchow2": ("./features/val/features_virchow2val_norm.npy"),
 "PLIP": ("./features/val/features_plipval_norm.npy"),
 "DINOv3": ("./features/val/features_dinov3val_norm.npy"),
 "CONCH": ("./features/val/features_conchval_norm.npy"),
 "Phikon2": ("./features/val/features_phikon2val_norm.npy")
}

classifiers = ["SVM", "Decision Tree", "Random Forest", "Red Propia"]
results= []

for name_feature, val_file in features.items():
   print(f"Using the feature extractor {name_feature}")
   if not os.path.exists(val_file):
      print(f"Validation file is missing for {name_feature}")
      continue

   X_val = np.load(val_file)
   
   if X_val.shape[0] != len(y_val):
      print("ERROR: The shapes are differents")
      continue
   
   scaler_path = f"./trained_models/scaler_{name_feature}_norm.joblib"
   if not os.path.exists(scaler_path):
    print(f"{scaler_path} not found")
    continue
   scaler = joblib.load(scaler_path)
   X_val_s = scaler.transform(X_val)
   
   for name_classifier in classifiers:
      print(f"Using the classifier {name_classifier}")
      if name_classifier=="Red Propia":
        path_model = f"./trained_models/{name_classifier}_{name_feature}_norm.keras"
        if not os.path.exists(path_model):
            print(f"{path_model} not found")
            continue
        model = tf.keras.models.load_model(path_model)
        y_prob = model.predict(X_val_s, verbose=0)
        y_prob_binary = y_prob[:, 1:].sum(axis=1)
        y_pred_binary = (y_prob_binary > 0.5).astype(int)
      else:
        path_model = f"./trained_models/{name_classifier}_{name_feature}_norm.joblib"
        if not os.path.exists(path_model):
            print(f"{path_model} not found")
            continue
        model = joblib.load(path_model)
        y_pred = model.predict(X_val_s)
        y_prob = model.predict_proba(X_val_s)
        y_pred_binary = binary(y_pred)
        y_prob_binary = y_prob[:, 1:].sum(axis=1)

      accuracy = accuracy_score(y_val, y_pred_binary)
      precision = precision_score(y_val, y_pred_binary, average='binary', zero_division=0)
      recall = recall_score(y_val, y_pred_binary, average='binary', zero_division=0)
      f1 = f1_score(y_val, y_pred_binary, average='binary', zero_division=0)
      kappa = cohen_kappa_score(y_val, y_pred_binary, weights=None)
  
      try:
         roc_auc = roc_auc_score(y_val, y_prob_binary)
      except ValueError:
         roc_auc = np.nan
      results.append({ 
       "Feature": name_feature,
       "Classifier": name_classifier,
       "Accuracy": round(accuracy,5),
       "F1 Score": round(f1,5),
       "Precision": round(precision, 5),
       "Recall": round(recall, 5),
       "ROC AUC": round(roc_auc, 5),
       "Cohen Kappa": round(kappa,5)
      })

df_results = pd.DataFrame(results)
output="SICAPV2_binary_norm.xlsx"
df_results.to_excel(output, index=False)
print("Finished")

