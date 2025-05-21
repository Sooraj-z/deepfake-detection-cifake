# 🕵️ Deepfake Detection using CIFAKE Dataset

## 🧠 Overview
This project uses custom CNNs and pretrained models (ResNet18, VGG16) to detect deepfake face images from the public CIFAKE dataset. Model performance is evaluated using accuracy, confusion matrix, ROC and PR curves. Interpretability is added through Grad-CAM heatmaps.

## 📦 Tools & Frameworks
- Python
- PyTorch, torchvision
- CNN, ResNet18, VGG16
- Grad-CAM (torchcam)
- sklearn, seaborn, matplotlib

## 📈 Model Results

| Model     | Accuracy | AUC (ROC) | AUC (PR) |
|-----------|----------|-----------|----------|
| Custom CNN | 92.5%   | ~0.94     | ~0.94    |
| ResNet18   | 85.6%   | ~0.86     | ~0.86    |
| VGG16      | 89.8%   | ~0.90     | ~0.90    |

## 🔍 Evaluation Techniques
- Confusion Matrix  
- ROC Curve (AUC)  
- Precision-Recall Curve  
- Grad-CAM visualizations on fake and real images  

## 🔬 Grad-CAM Examples
![GradCAM](plots/gradcam_examples.png)

## 🧪 Dataset
**CIFAKE — Real and GAN-generated synthetic images**  
[🔗 View Dataset on HuggingFace](https://huggingface.co/datasets/utkarsh5123/cifake)

## 📁 Repo Contents
- `CIFAKE_Deep_Learning.ipynb` – full model pipeline (Colab notebook)
- `plots/` – confusion matrix, ROC, PR curves, Grad-CAM overlays
- `models/` – saved PyTorch weights
- `logs/` – training logs in `.pkl` format
