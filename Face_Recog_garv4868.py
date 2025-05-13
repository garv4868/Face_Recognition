import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# Load the dataset
def load_dataset(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        for img_path in os.listdir(os.path.join(data_dir, label)):
            img = cv2.imread(os.path.join(data_dir, label, img_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))
            X.append(img.flatten())
            y.append((label))
    return np.array(X), np.array(y)

X, y = load_dataset('dataset/faces')

# Preprocess the data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_zero_mean = (X - X_mean) / X_std

# Perform PCA
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_zero_mean)

# Train the model
clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, activation='relu')
clf.fit(X_pca, y)

# Test the model
def predict_face(model, img):
    img = cv2.resize(img, (100, 100))
    img = (img.flatten() - X_mean) / X_std
    img_pca = pca.transform([img])
    return model.predict(img_pca)[0]

# Example usage
img = cv2.imread('dataset/faces/Aamir/face_13.jpg', cv2.IMREAD_GRAYSCALE)
label = predict_face(clf, img)
print(f'Predicted label: {label}')