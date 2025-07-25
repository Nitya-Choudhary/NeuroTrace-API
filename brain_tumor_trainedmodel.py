# -*- coding: utf-8 -*-
"""Brain_tumor_trainedmodel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rMFO3yy6PtTj1bhs4-j63D05WujIayHq
"""

!pip install tensorflow matplotlib scikit-learn

from google.colab import drive
drive.mount('/content/drive')

from google.colab import files
    uploaded = files.upload()

import zipfile
import os

# Unzip the uploaded file (replace filename if needed)
zip_path = "/content/braintumor.zip"  # or use: list(uploaded.keys())[0]
extract_path = "/content/brain_tumor_dataset"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Check folders
for root, dirs, files in os.walk(extract_path):
    print(f"{root} --> {len(files)} files")

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join(extract_path, 'Brain MRI Images/Brain MRI Images/Train')
val_dir = os.path.join(extract_path, 'Brain MRI Images/Brain MRI Images/Validation')

img_size=(150,150)

train_gen= ImageDataGenerator(rescale=1./255)
val_gen=ImageDataGenerator(rescale=1./255)

train_data= train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary'
)

val_data= val_gen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='binary'
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten,Dense , Dropout

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, epochs=10, validation_data=val_data)

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'],label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()

model.save("brain_tumor_model.h5")

from google.colab import files
files.download("brain_tumor_model.h5")