import pandas as pd
import numpy as np
import os

df_labels = pd.read_csv('sicap_splits.csv')
df_labels = df_labels.sort_values('image_name').reset_index(drop=True)
df_train = df_labels[df_labels['partition'] == 'train']
df_test = df_labels[df_labels['partition'] == 'test']

y_train = df_train['label'].values
y_test = df_test['label'].values

print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de y_test: {y_test.shape}")

df_train.to_csv('train.csv', index=False)
df_test.to_csv('val.csv', index=False)

