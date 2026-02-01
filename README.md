# Alzheimer’s Disease Detection Using CNN

## Project Overview
Alzheimer’s disease is a progressive neurological disorder that affects memory, thinking, and behavior. Early diagnosis is crucial for effective treatment and patient care.

This project focuses on building a Convolutional Neural Network (CNN) model to classify MRI brain scans into different stages of Alzheimer’s disease using deep learning techniques.

---

## Objectives
- Classify MRI images into Alzheimer’s disease categories  
- Build an end-to-end deep learning pipeline  
- Handle image data stored in Parquet format  
- Evaluate model performance using medical metrics  
- Understand real-world challenges in medical AI  

---

## Dataset
- MRI brain scan data stored in Parquet (.parquet) files  
- Images are stored as byte arrays inside the dataset  
- Labels provided for multi-class classification  

### Classes
- Non Demented  
- Very Mild Demented  
- Mild Demented  
- Moderate Demented  

(Dataset used for academic and learning purposes)

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- OpenCV  
- Matplotlib  
- TensorFlow / Keras  
- Scikit-learn  

---

## Data Preprocessing
The following preprocessing steps were applied:

- Reading Parquet files using Pandas  
- Extracting image bytes from Parquet columns  
- Converting byte data into pixel arrays  
- Image normalization  
- Resizing images to a fixed resolution (e.g., 128 × 128)  
- Channel handling for CNN compatibility  
- Label encoding  
- Train–test split  

---

## Model Architecture
A transfer learning CNN model was implemented using VGG19 pretrained on ImageNet as the feature extractor. The architecture includes:

-VGG19 convolutional base for feature extraction
-The last 5 convolutional layers were fine-tuned to adapt to the target dataset
-Global Average Pooling (GAP) layer to reduce the spatial dimensions and prevent overfitting
-Dense layer (256 units, ReLU activation) for learning high-level features
-Dropout layer (0.5) to reduce overfitting
-Softmax output layer for multi-class classification 

---

## Model Evaluation
The model was evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

### Results

| Metric | Value |
|------|------|
| Training Accuracy | 0.96 |
| Validation Accuracy | 0.89 |
| Test Accuracy | 0.87 |

---

## Classification Report
A detailed classification report was generated to analyze class-wise performance, which is especially important in medical diagnosis tasks where recall plays a critical role.
| No | Precision | Recall | F1-score | Support |
|----|-----------|--------|----------|---------|
| 0  | 0.99      | 0.78   | 0.87     | 172     |
| 1  | 0.76      | 0.87   | 0.81     | 15      |
| 2  | 0.83      | 0.98   | 0.90     | 634     |
| 3  | 0.93      | 0.76   | 0.84     | 459     |

---

## Challenges
- Medical images stored as byte arrays in Parquet format  
- Class imbalance across disease categories  
- High computational cost of CNN training  
- Risk of overfitting due to limited MRI samples  

---

## Future Improvements  
- Implement Grad-CAM for model explainability  
- Improve recall for minority disease classes  
- Apply advanced data augmentation techniques  
- Deploy the model using Streamlit or Flask  

---

## Conclusion
This project demonstrates the application of deep learning for medical image classification using MRI scans stored in Parquet format. The CNN model learns meaningful spatial patterns from brain images and shows promising performance for Alzheimer’s disease detection. Further improvements using explainable AI and transfer learning can enhance clinical reliability.

---

---
