# -*- coding: utf-8 -*-

import spectral
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
from ml_functions import mask_multiplication, count_pixels, read_pixels, predict_tissue_fast

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report

# dostępne funkcje
#print(dir(spectral))
masks_path='/media/sonia/8E6C6F3D6C6F1EE9/Solaris/ML_git/'
classes=['cancer.bmp','benign_tumor.bmp', 'stroma_all.bmp']

#classes_colors=((255,0,0),(0,255,0),(128,128,128))
classes_colors = (
    '#e63946',  # cancer – czerwony
    '#f0901a',  # benign – pomarańcz
    '#2a9d8f'   # stroma – zielony
)

tissue_mask = "tissue_mask.bmp" 

mask_multiplication(masks_path, classes, tissue_mask)

indexes, M = count_pixels(masks_path, classes)

#Otwarcie pliku z metrykami
metrics = spectral.open_image("BR2085d_3_5_metrics_new_MNF20.hdr")
print('metrics type', type(metrics))
#Size of file
num_bands = metrics.shape[2]
num_rows = metrics.shape[0]
num_columns = metrics.shape[1]


#Read data
data, labels = read_pixels(indexes,M,metrics)

#plt.plot(data[20,:])
#plt.show()

#Data split for training and classification 

# Split the dataset into training and testing sets
Data_train, Data_test, label_train, label_test = train_test_split(data, labels, test_size=0.5)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(Data_train, label_train)

# Make predictions on the test data
label_pred = rf_classifier.predict(Data_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(label_test, label_pred)
classification_rep = classification_report(label_test, label_pred)

# Print the results
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)


predict_tissue_fast(rf_classifier, masks_path, tissue_mask, metrics, classes_colors)



