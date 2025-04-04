# Image Processing and Deep Learning for Animal Classification

This project implements an end-to-end pipeline for image classification using both traditional machine learning and deep learning techniques. The workflow includes steps for library installation, data loading, exploratory data analysis, feature extraction, model training, evaluation, and final predictions.

---

## 1. Environment Setup

- **Library Installation:**  
  Installs packages such as `requests`, `opencv-python`, `tqdm`, `scikit-image`, `seaborn`, `matplotlib`, `scikit-learn`, and `xgboost`.

- **Imports and Configuration:**  
  Imports essential modules for image processing (OpenCV, scikit-image), machine learning (scikit-learn, XGBoost), and deep learning (TensorFlow, Keras).  
  Sets a random seed for reproducibility and mounts Google Drive to access datasets.

---

## 2. Data Loading and Preparation

- **Dataset Loading:**  
  Reads CSV files (e.g., `train.csv` and `test.csv`) containing image IDs and labels from a specified folder in Google Drive.

- **Image Loading:**  
  Uses a custom function to load and resize images (e.g., to 200×200 pixels) from the train and test directories.

---

## 3. Exploratory Data Analysis (EDA)

- **Dataset Overview:**  
  Provides summaries including the number of entries, shape, and basic statistics.  
  Visualizes class distribution using count plots, showing balanced classes (e.g., cats vs. dogs).

- **Image Inspection:**  
  Displays sample images for each class.  
  Plots histograms of image dimensions and pixel intensities to verify consistency and assess image quality.

- **Feature Visualization:**  
  Extracts and visualizes Histogram of Oriented Gradients (HOG) features and analyzes brightness and contrast variations.

- **Dimensionality Reduction:**  
  Applies PCA and t-SNE on flattened image features to visualize the feature space and understand class separability.

---

## 4. Feature Extraction

- **Traditional Feature Extraction Methods:**  
  - **Grayscale Flattening:** Directly flattens grayscale images into vectors.
  - **Smoothing:** Applies Gaussian blur before flattening.
  - **Edge Detection:** Uses Canny and Sobel methods to capture edge information.
  - **HOG Features:** Extracts detailed edge and texture descriptors.
  - **Adaptive Histogram Equalization:** Enhances image contrast before feature extraction.
  - **Feature Combination:** Concatenates multiple feature sets (e.g., grayscale, HOG, and edge detection features).

---

## 5. Machine Learning Models

- **Model Training and Evaluation:**  
  The pipeline tests several classifiers (SVM, Decision Tree, Random Forest) on features extracted using different methods.  
  Custom functions split data, train models, and evaluate accuracy with confusion matrices.

- **Hyperparameter Tuning and Ensemble Methods:**  
  Uses grid search and randomized search to optimize model parameters.  
  Implements a Voting Classifier to combine predictions from different models for improved performance.

---

## 6. Deep Learning with Transfer Learning

- **Data Augmentation:**  
  Uses `ImageDataGenerator` to augment images (e.g., rotations, shifts, zooming, horizontal flips) and improve generalization.

- **Transfer Learning with MobileNetV2:**  
  Loads MobileNetV2 (pre-trained on ImageNet) as the base model and freezes its layers.  
  Adds custom layers (global average pooling, dense layers with dropout and batch normalization) on top for classification.

- **Model Training:**  
  Compiles the CNN using the Adam optimizer and binary cross-entropy loss.  
  Trains the model with callbacks (learning rate scheduler, early stopping, and reduce LR on plateau) and visualizes training curves.

- **Test Predictions:**  
  Preprocesses and resizes test images, predicts labels with the trained model, and saves the submission as a CSV file.

---

## Conclusion

This project demonstrates a robust approach for image classification through both traditional feature engineering and state-of-the-art deep learning. The modular design allows for experimentation with different feature extraction methods and model architectures, making it adaptable to various image classification challenges.


