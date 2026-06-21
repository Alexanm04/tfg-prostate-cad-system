import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping

class MiRed:
   def __init__(self, input_dimension, epochs=50, batch_size=32):
      self.epochs=epochs
      self.batch_size=batch_size

      self.model=Sequential([
         Input(shape=(input_dimension,)),
         Dense(1024, activation='relu'),
         BatchNormalization(),
         Dropout(0.4),
         Dense(512,activation='relu'),
         BatchNormalization(),
	 Dropout(0.3),
         Dense(256, activation='relu'),
         BatchNormalization(),
         Dropout(0.3),
         Dense(128, activation='relu'),
         BatchNormalization(),
         Dropout(0.3),
         Dense(4, activation='softmax')
      ])
      
      self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

   def fit(self, X, y, class_weight= None):
      early_s = EarlyStopping(
         monitor='loss',
         patience=5,
         restore_best_weights=True
      )
      self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, callbacks=[early_s], class_weight=class_weight)

   def predict(self, X):
      probabilities = self.model.predict(X, verbose=0)
      predicted_classes = np.argmax(probabilities, axis=1)
      return predicted_classes

   def predict_proba(self, X):
      return self.model.predict(X, verbose=0)
