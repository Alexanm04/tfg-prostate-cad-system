import os 
import glob
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

wsi_features_dir = "./features/test_wsi"
name_feature = "CONCH"
classifiers =["SVM", "Red Propia", "Random Forest"]

for name_classifier in classifiers:
    print(f"Clasificador {name_classifier}")
    scaler_path = f"./trained_models/scaler_{name_feature}_norm.joblib"
    scaler = joblib.load(scaler_path)
    if (name_classifier == "Red Propia"):
        path_model = f"./trained_models/{name_classifier}_{name_feature}_norm.keras"
        model = tf.keras.models.load_model(path_model)
    else:
        path_model = f"./trained_models/{name_classifier}_{name_feature}_norm.joblib"
        model = joblib.load(path_model)

    csv_files = sorted(glob.glob(f"{wsi_features_dir}/*_labels_and_coords.csv"))
    for csv_file in csv_files:
        print(f"Using CSV {csv_file}")
        npy_file = csv_file.replace('_labels_and_coords.csv', f'_features_{name_feature.lower()}.npy')
        if not os.path.exists(npy_file):
            print(f"Test file is missing for {npy_file}")
            continue
        
        df = pd.read_csv(csv_file)
        X_wsi = np.load(npy_file)
        if X_wsi.shape[0] != len(df):
            print("ERROR: The shapes are differents")
            continue
        
        X_wsi_s = scaler.transform(X_wsi)

        if (name_classifier == "Red Propia"):
            y_probability = model.predict(X_wsi_s, verbose = 0)
            y_prediction = np.argmax(y_probability, axis = 1)
        else:
            y_prediction = model.predict(X_wsi_s)
            y_probability = model.predict_proba(X_wsi_s)

        df['predicted_label'] = y_prediction
        df['prediction_confidence'] = np.max(y_probability, axis=1)
        df['0_probability'] = y_probability[:, 0]
        df['1_probability'] = y_probability[:, 1]
        df['2_probability'] = y_probability[:, 2]
        df['3_probability'] = y_probability[:, 3]
        safe_classifier = name_classifier.replace(' ', '_')
        output = csv_file.replace('_labels_and_coords.csv', f'_labels_and_coords_and_predictions_{name_feature}_{safe_classifier}.csv')
        df.to_csv(output, index=False)
print("Finished")
