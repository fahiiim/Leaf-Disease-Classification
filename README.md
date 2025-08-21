# Agri-Tech: Leaf Disease Classification

This notebook demonstrates a comprehensive pipeline for classifying plant leaf diseases using deep learning, specifically an EfficientNetB1 model. Below is a structured overview that follows the workflow and logic of the notebook's content.

---

## 1. Imports and Setup

The notebook uses a variety of libraries for data manipulation (`numpy`, `pandas`), visualization (`matplotlib`, `seaborn`), image processing (`cv2`, `PIL`), deep learning (`tensorflow`, `keras`), and machine learning utilities (`sklearn`). It sets up plotting styles and configures the environment for Jupyter.

---

## 2. Data Loading and Inspection
- **Dataset Link:**
  Link - https://www.kaggle.com/datasets/emmarex/plantdisease/data
- **Sample Image Display:**  
  Loads and displays a sample image from the dataset to verify shape and format.

- **Building the DataFrame:**  
  Iterates through the dataset directory, collecting image file paths and their class labels into a DataFrame with columns `filepaths` and `labels`.

- **Dataset Summary:**  
  Prints the number of samples per class, revealing a significant class imbalance.
  <img width="1252" height="1194" alt="image" src="https://github.com/user-attachments/assets/46eb89d1-8e95-4986-87a2-1c013b9f361f" />


---

## 3. Data Balancing and Augmentation

- **Class Balancing:**  
  Limits the maximum number of samples per class to 500 by random sampling, reducing overrepresented classes.

- **Directory Setup for Augmented Images:**  
  Creates an `aug` directory structure to store augmented images for underrepresented classes.

- **Data Augmentation:**  
  Uses Keras' `ImageDataGenerator` to generate new images for classes with fewer than 500 samples, applying transformations like horizontal flip, rotation, shift, and zoom.

- **Augmentation Summary:**  
  Lists the count of augmented images generated per class.

- **Visualizing Augmented Images:**  
  Displays a grid of randomly augmented images.

- **Augmented DataFrame Construction:**  
  Combines augmented images with the balanced dataset to ensure all classes have equal representation.

---

## 4. Dataset Splitting

- **Train/Validation/Test Split:**  
  Divides the dataset into training (80%), validation (10%), and test (10%) sets using `train_test_split`.

---

## 5. Data Generators

- **Image Generators:**  
  Prepares Keras `ImageDataGenerator` objects for training, validation, and test sets, including preprocessing and augmentation for the training images.

- **Class List and Steps:**  
  Extracts the list of classes and calculates steps per epoch for training and testing.

- **Sample Visualization:**  
  Defines a function to visualize batches of images with their labels from the generator.

---

## 6. Model Architecture

- **EfficientNetB1:**  
  Loads pre-trained EfficientNetB1 without top layers, adds batch normalization, a dense layer with regularization, dropout, and a final softmax layer matching the number of classes.

- **Compilation:**  
  Compiles the model with Adamax optimizer and categorical crossentropy loss.

---

## 7. Custom Callback for Training Control

- **Learning Rate Adjustment (LRA) Callback:**  
  Custom callback to dynamically adjust the learning rate based on training and validation performance, with early stopping and best weight restoration.

---

## 8. Model Training

- **Training Setup:**  
  Configures epochs, patience, thresholds, and other hyperparameters.

- **Training Execution:**  
  Trains the model with the custom callback. Training can be interactively halted by the user after a certain number of epochs.

---

## 9. Results Visualization

- **Training Curves:**  
  Function to plot training and validation loss and accuracy across epochs, highlighting the best epoch.
  <img width="1301" height="541" alt="image" src="https://github.com/user-attachments/assets/aed9cfed-c6d0-46a4-ad62-c83193b0ddc8" />
-  **Confusion Matrix:**
  <img width="810" height="801" alt="image" src="https://github.com/user-attachments/assets/3eee6236-8298-4cd6-9b7e-3d1d832fab2e" />

---

## 10. Evaluation and Reporting

- **Model Evaluation:**  
  Evaluates performance on the test set and prints the test accuracy.

- **Saving Model:**  
  Saves the trained model with a filename reflecting its accuracy.

- **Predictions and Reports:**  
  - Makes predictions on the test set.
  - Generates and displays a confusion matrix and a detailed classification report (precision, recall, f1-score, support for each class).

---

## 11. Inference: Predicting New Images

- **Single Image Prediction:**  
  Demonstrates how to preprocess and predict the class of a single image, including resizing, expanding dimensions, and interpreting the output probabilities.

- **Batch Image Prediction:**  
  Shows how to batch-predict a small set of images from a directory, outputting the predicted class and probability for each.

---

## 12. Key Results

- **High Accuracy:**  
  The model achieves approximately 99% accuracy on the test set, with strong performance across all classes as shown in the classification report.

---

## 13. How to Use

- **Inference Instructions:**  
  Example code is provided for both single and batch image prediction, making the notebook practical for real-world use after training.

---

**Summary:**  
This notebook covers the end-to-end process of image-based plant disease classification, including data loading, balancing, augmentation, deep learning model training, evaluation, and deployment. The approach is robust and achieves high accuracy, making it suitable for application in agricultural diagnostics..
