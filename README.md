# Crop Disease Detection Using Leaf Images

## 1. Introduction
Agriculture is a primary sector that significantly contributes to food security and economic development. However, crop diseases adversely affect yield quality and productivity. Traditional disease diagnosis relies on expert observation, which may be time-consuming and prone to errors. With advancements in Artificial Intelligence, particularly Deep Learning, it is now possible to automatically detect plant diseases using leaf images.

This project proposes a Convolutional Neural Network (CNN) based classification model for diagnosing plant diseases from leaf images. A user-friendly Streamlit web application is integrated to make the system easily accessible to farmers, students, and researchers.

## 2. Problem Statement
- Farmers often fail to detect crop diseases at an early stage.
- Manual disease identification requires expert knowledge and is not scalable.
- Delay in diagnosis leads to reduced crop productivity and financial loss.

Therefore, an automated, fast, and reliable plant disease detection system is required.

## 3. Objectives
| Objective | Description |
|---------|-------------|
| Disease Identification | To classify leaf images as healthy or diseased. |
| Deep Learning Model | To train a CNN model using the PlantVillage dataset. |
| Real-Time Prediction | To develop a web interface for disease detection. |
| Accessibility | To assist farmers with a simple and affordable solution. |

## 4. Dataset Description
- Dataset Used: PlantVillage Dataset
- Source: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- Total Images: ~54,000
- Classes: Multiple disease types and healthy leaves

### Data Preprocessing
- Image resizing (224 × 224)
- Normalization
- Data Augmentation (rotation, flip, zoom)
- Train/Validation/Test split

## 5. Methodology

### System Architecture
Dataset → Preprocessing → CNN Model Training → Model Evaluation → Streamlit Web App

### CNN Model Components
- Convolution Layers (Feature Extraction)
- Max Pooling Layers (Dimensionality Reduction)
- Fully Connected Layers (Classification)
- Softmax Output Layer

### Training Parameters
| Parameter | Value |
|---------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | 25–50 |

## 6. Implementation

### Requirements Installation
```
pip install -r requirements.txt
```

### Running the Web Application
```
streamlit run app.py
```

### Usage Instructions
1. Upload a leaf image through the interface.
2. The model processes the image.
3. The predicted disease class and confidence score are displayed.

## 7. Results & Performance

| Metric | Score |
|--------|-------|
| Training Accuracy | ~95-98% |
| Validation Accuracy | ~90-95% |

## 8. Conclusion
This project demonstrates a cost-effective and automated disease diagnosis approach using deep learning. The system is efficient, user-friendly, and beneficial for early disease detection and prevention of crop loss.

## 9. Future Enhancements
| Proposed Improvement | Benefit |
|---------------------|---------|
| Deploy to Cloud | Access from anywhere |
| Add Disease Treatment Suggestions | Practical farmer guidance |
| Convert to Mobile App (TFLite) | On-field usage |
| Expand Dataset | Increase robustness |

## 10. References
1. PlantVillage Dataset, Kaggle
2. TensorFlow Documentation
3. Chollet, F. Deep Learning with Python, Manning
4. Agricultural Disease Analysis Research Studies

## 11. Author
**Jaswanth Badipati**  
LinkedIn: https://linkedin.com/in/jaswanthbadipati  
GitHub: https://github.com/jaswanthbadipati

**Sai Balaji Chimata**  
LinkedIn: https://www.linkedin.com/in/sai-balaji-chimata-3791b32a5/

**Tarun Utukuri**  
LinkedIn: [https://linkedin.com/in/tarunutukuri  ](https://www.linkedin.com/in/tarun9905/)
