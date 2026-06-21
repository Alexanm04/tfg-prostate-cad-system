import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix)
from red_propia import MiRed
from sklearn.utils.class_weight import compute_class_weight
import joblib
import tensorflow as tf
df_train = pd.read_csv('train.csv')
df_val = pd.read_csv('val.csv')

y_train = df_train['label'].values
y_val = df_val['label'].values

features = {
 "ResNet-50": ("./features/train/features_resnet50train.npy", "./features/val/features_resnet50val.npy"),
 "UNI2-h": ("./features/train/features_uni2train.npy", "./features/val/features_uni2val.npy"),
 "Prov-GigaPath": ("./features/train/features_gigapathtrain.npy", "./features/val/features_gigapathval.npy"),
 "Virchow2": ("./features/train/features_virchow2train.npy", "./features/val/features_virchow2val.npy"),
 "PLIP": ("./features/train/features_pliptrain.npy", "./features/val/features_plipval.npy"),
 "DINOv3": ("./features/train/features_dinov3train.npy", "./features/val/features_dinov3val.npy"),
 "CONCH": ("./features/train/features_conchtrain.npy", "./features/val/features_conchval.npy"),
 "Phikon2": ("./features/train/features_phikon2train.npy", "./features/val/features_phikon2val.npy")
}

results= []

for name_feature, (train_file, val_file) in features.items():
   print(f"Using the feature extractor {name_feature}")
   if not os.path.exists(train_file) or not os.path.exists(val_file):
      print(f"Train file or validation file is missing for {name_feature}")
      continue

   X_train = np.load(train_file)
   X_val = np.load(val_file)
   
   if X_train.shape[0] != len(y_train) or X_val.shape[0] != len(y_val):
      print("ERROR: The shapes are differents")
      continue
   
   classifiers = {
    "SVM": CalibratedClassifierCV(
       estimator=LinearSVC(class_weight='balanced', random_state=42, dual=False, max_iter=5000),
       cv=5,
       n_jobs=-1
    ),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
    "Red Propia": MiRed(input_dimension=X_train.shape[1], epochs=50)
   }

   scaler = StandardScaler()
   X_train_s = scaler.fit_transform(X_train)
   X_val_s = scaler.transform(X_val)

   joblib.dump(scaler, f"./trained_models/scaler_{name_feature}.joblib")

   unique_classes = np.unique(y_train)
   weights = compute_class_weight(class_weight='balanced', classes= unique_classes, y=y_train)
   weight_dictionary = dict(zip(unique_classes, weights))
   
   for name_classifier, classifier in classifiers.items():
      print(f"Using the classifier {name_classifier}")
      if name_classifier=="Red Propia":
         classifier.fit(X_train_s, y_train, class_weight=weight_dictionary)
         path_model = f"./trained_models/{name_classifier}_{name_feature}.keras"
         classifier.model.save(path_model)
      else:
         classifier.fit(X_train_s, y_train)
         joblib.dump(classifier, f"./trained_models/{name_classifier}_{name_feature}.joblib")
         
      y_prediction = classifier.predict(X_val_s)
      y_probability= classifier.predict_proba(X_val_s)
      accuracy = accuracy_score(y_val, y_prediction)
      precision = precision_score(y_val, y_prediction, average='macro', zero_division=0)
      recall = recall_score(y_val, y_prediction, average='macro', zero_division=0)
      f1 = f1_score(y_val, y_prediction, average='macro', zero_division=0)
      kappa = cohen_kappa_score(y_val, y_prediction, weights='quadratic')
  
      try:
         roc_auc = roc_auc_score(y_val, y_probability, multi_class='ovr', average='macro')
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
output="SICAPV2_trained_classifiers.xlsx"
df_results.to_excel(output, index=False)
print("Finished")

