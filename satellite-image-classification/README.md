# AI Engineer - Satellite Image Classification

[![IBM AI Engineering](https://img.shields.io/badge/IBM-AI%20Engineering-blue)](https://www.coursera.org/professional-certificates/ai-engineer)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)](https://pytorch.org)

This repository contains my project work developed as part of the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera.

## Description

A comprehensive deep learning project for satellite image classification using both Keras/TensorFlow and PyTorch frameworks. The project covers the full workflow for classifying agricultural vs non-agricultural land from satellite imagery using Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and hybrid CNN-ViT models.

## 📋 Quick Navigation

- [🎯 Project Overview](#-project-overview)
- [📁 Repository Structure](#-repository-structure)
- [🚀 Getting Started](#-getting-started)
- [🔬 Notebooks Overview](#-notebooks-overview)
- [🏗️ Model Architectures](#️-model-architectures)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [📚 Learning Objectives](#-learning-objectives)

## 🎯 Project Overview

This project demonstrates the complete machine learning pipeline for satellite image classification, comparing different approaches and frameworks:

- **Data Loading Strategies**: Memory-based vs generator-based data loading
- **Deep Learning Frameworks**: Keras/TensorFlow and PyTorch implementations
- **Model Architectures**: CNN, Vision Transformers, and CNN-ViT hybrid models
- **Performance Evaluation**: Comprehensive metrics and comparative analysis

### 🔑 Highlights

- Comparative analysis between TensorFlow/Keras and PyTorch implementations  
- Implementation of CNNs, Vision Transformers, and CNN-ViT hybrid models  
- Demonstration of data loading, augmentation, and evaluation techniques  
- Reproducible experiments with fixed random seeds  

## 📁 Repository Structure

```
ai-engineer/
├── satellite-image-classification/
│   ├── 📊 Data Loading & Preparation
│   │   ├── 01 Data Loading - Memory vs Generator.ipynb
│   │   ├── 02k Data Preparation - Keras.ipynb
│   │   └── 02p Data Preparation - PyTorch.ipynb
│   │
│   ├── 🤖 Model Development
│   │   ├── 03k Model Development and Evaluation - Keras.ipynb
│   │   └── 03p Model Development and Evaluation - PyTorch.ipynb
│   │
│   ├── 📈 Evaluation & Comparison
│   │   ├── 04 Evaluation - Keras and PyTorch Models.ipynb
│   │   ├── 05k CNN and Vision Transformers Hybrid - Keras.ipynb
│   │   ├── 05p CNN and Vision Transformers Hybrid - PyTorch.ipynb
│   │   └── 06 Evaluation - CNN and Vision Transformers Hybrids.ipynb
│   │
│   ├── 🗂️ Model Files (Local Only)
│   │   ├── *.keras                            # Keras model files (excluded from git)
│   │   ├── *.pth                              # PyTorch model files (excluded from git)
│   │   └── images_dataSAT/                    # Dataset (excluded from git)
│   │       ├── class_0_non_agri/              # Non-agricultural images (3,000 files)
│   │       └── class_1_agri/                  # Agricultural images (3,000 files)
│   │
│   └── 📚 Documentation
│       └── docs/                              # Additional documentation
│
├── README.md                                  # This file
├── .gitignore                                 # Git ignore rules
└── venv/                                      # Virtual environment (excluded from git)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook/Lab
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-engineer
   ```  

2.	**Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```  

3.	**Install dependencies**
   ```bash
   pip install tensorflow keras torch torchvision
   pip install jupyter matplotlib seaborn pandas numpy
   pip install scikit-learn pillow opencv-python
   ```  

4.	**Launch Jupyter**
   ```bash
   jupyter lab
   ```  
   
## 📊 Dataset

The project uses a satellite image dataset with two classes:
- **Class 0**: Non-agricultural land (3,000 images)
- **Class 1**: Agricultural land (3,000 images)

Images are stored in the `satellite-image-classification/images_dataSAT/` directory.

> **Note**: The dataset and model files are excluded from git due to size constraints but are present locally for development.

## 📚 Learning Objectives

After completing this project, you will be able to:

1. **Data Management**: Efficiently handle large satellite image datasets
2. **Model Development**: Build and train CNN and ViT models from scratch
3. **Framework Proficiency**: Work with both Keras/TensorFlow and PyTorch
4. **Advanced Architectures**: Implement and understand hybrid CNN-ViT models
5. **Performance Analysis**: Evaluate models using multiple metrics and visualizations
6. **Comparative Analysis**: Compare different approaches and frameworks

## 🔬 Notebooks Overview

### 📊 **Data Loading & Preparation**
| Notebook | Description | Framework |
|:---------|:------------|:----------|
| [`01 Data Loading - Memory vs Generator.ipynb`](satellite-image-classification/01%20Data%20Loading%20-%20Memory%20vs%20Generator.ipynb) | Compare memory-based vs generator-based data loading strategies | Both |
| [`02k Data Preparation - Keras.ipynb`](satellite-image-classification/02k%20Data%20Preparation%20-%20Keras.ipynb) | Keras-specific preprocessing and augmentation | Keras/TensorFlow |
| [`02p Data Preparation - PyTorch.ipynb`](satellite-image-classification/02p%20Data%20Preparation%20-%20PyTorch.ipynb) | PyTorch-specific preprocessing and augmentation | PyTorch |

### 🤖 **Model Development**
| Notebook | Description | Framework |
|:---------|:------------|:----------|
| [`03k Model Development and Evaluation - Keras.ipynb`](satellite-image-classification/03k%20Model%20Development%20and%20Evaluation%20-%20Keras.ipynb) | CNN model development and training | Keras/TensorFlow |
| [`03p Model Development and Evaluation - PyTorch.ipynb`](satellite-image-classification/03p%20Model%20Development%20and%20Evaluation%20-%20PyTorch.ipynb) | CNN model development and training | PyTorch |

### 📈 **Evaluation & Advanced Architectures**
| Notebook | Description | Framework |
|:---------|:------------|:----------|
| [`04 Evaluation - Keras and PyTorch Models.ipynb`](satellite-image-classification/04%20Evaluation%20-%20Keras%20and%20PyTorch%20Models.ipynb) | Performance comparison between frameworks | Both |
| [`05k CNN and Vision Transformers Hybrid - Keras.ipynb`](satellite-image-classification/05k%20CNN%20and%20Vision%20Transformers%20Hybrid%20-%20Keras.ipynb) | CNN-ViT hybrid model implementation | Keras/TensorFlow |
| [`05p CNN and Vision Transformers Hybrid - PyTorch.ipynb`](satellite-image-classification/05p%20CNN%20and%20Vision%20Transformers%20Hybrid%20-%20PyTorch.ipynb) | CNN-ViT hybrid model implementation | PyTorch |
| [`06 Evaluation - CNN and Vision Transformers Hybrids.ipynb`](satellite-image-classification/06%20Evaluation%20-%20CNN%20and%20Vision%20Transformers%20Hybrids.ipynb) | Final evaluation of hybrid models | Both |

## 🏗️ Model Architectures

### 1. **Convolutional Neural Networks (CNNs)**
- **Local feature extraction** - Excellent at capturing spatial patterns
- **Efficient for smaller datasets** - Requires less data than transformers
- **Fast training** - Optimized for image classification tasks

### 2. **Vision Transformers (ViTs)**
- **Self-attention mechanisms** - Global spatial dependencies
- **Strong performance** - Superior on complex vision tasks
- **Scalable** - Performance improves with more data

### 3. **CNN-ViT Hybrid Models**
- **Best of both worlds** - Combines CNN's local features with ViT's global attention
- **Efficient and high-performing** - Optimal balance of speed and accuracy
- **Feature extraction pipeline** - CNN extracts local features, ViT processes global patterns

## 📈 Evaluation Metrics

| Metric | Description | Purpose |
|:-------|:------------|:--------|
| **Accuracy** | Overall correctness of predictions | General model performance |
| **Precision** | Correctness of positive predictions | False positive control |
| **Recall** | Ability to detect positive cases | False negative control |
| **F1-Score** | Harmonic mean of precision & recall | Balanced performance measure |
| **ROC Curves** | Threshold analysis | Optimal threshold selection |
| **Confusion Matrices** | Detailed classification results | Error analysis |

## 🛠️ Key Features

- **🔄 Framework Comparison**: Side-by-side Keras and PyTorch implementations
- **⚡ Memory Efficiency**: Generator-based loading for large datasets
- **🧠 Advanced Architectures**: CNN-ViT hybrid model implementation
- **🔬 Reproducible Results**: Fixed random seeds for consistent experiments
- **📊 Comprehensive Evaluation**: Multiple metrics and visualization tools

## 📖 Usage

1. **Start with data loading** - Begin with notebook `01` to understand the dataset
2. **Follow sequential workflow** - Run notebooks in numerical order (01-06)
3. **Framework-specific paths** - Choose Keras (`k`) or PyTorch (`p`) versions
4. **Experiment freely** - Modify hyperparameters and architectures
5. **Compare results** - Use evaluation notebooks to analyze performance

## 🤝 Contributing

This project is part of an AI Engineer learning path. Feel free to:

- **Experiment** with new model architectures
- **Add** evaluation metrics and visualizations
- **Improve** preprocessing techniques
- **Share** insights and results

---

## 🙏 Acknowledgements

- This project was completed as part of the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera
- Special thanks to the instructors and the Coursera team for providing excellent course materials
- The course exercises and instructions are provided by Coursera/IBM
- This repository contains my personal solutions, code implementations, and notes created while completing the course

## 📄 License & Disclaimer

This project is provided for **educational purposes only** as part of the AI Engineer learning curriculum. The code, models, and notebooks are shared to demonstrate skills in deep learning, computer vision, and machine learning engineering.

### ⚠️ Important Notice

- **Course materials and instructions** are provided by Coursera/IBM
- **This repository** contains only my personal solutions, code implementations, and notes
- **All course materials** and related intellectual property remain the copyright of Coursera and IBM
- **Not intended for production use** - Educational demonstration only
