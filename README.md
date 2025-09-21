# AI Engineer - Satellite Image Classification

This repository contains my project work developed as part of the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera.  

A comprehensive deep learning project for satellite image classification using both Keras/TensorFlow and PyTorch frameworks. The project covers the full workflow for classifying agricultural vs non-agricultural land from satellite imagery using Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and hybrid CNN-ViT models.

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

ai-engineer/
├── satellite-image-classification/
│   ├── 01 Data Loading - Memory vs Generator.ipynb
│   ├── 02k Data Preparation - Keras.ipynb
│   ├── 02p Data Preparation - PyTorch.ipynb
│   ├── 03k Model Development and Evaluation - Keras.ipynb
│   ├── 03p Model Development and Evaluation - PyTorch.ipynb
│   ├── 04 Evaluation - Keras and PyTorch Models.ipynb
│   ├── 05k CNN and Vision Transformers Hybrid - Keras.ipynb
│   ├── 05p CNN and Vision Transformers Hybrid - PyTorch.ipynb
│   ├── 06 Evaluation - CNN and Vision Transformers Hybrids.ipynb
│   ├── dev/                                    # Development files and saved models
│   │   ├── *.keras                            # Keras model files
│   │   ├── *.pth                              # PyTorch model files
│   │   └── *.ipynb                            # Additional lab notebooks
│   └── image_dataSAT/                         # Dataset directory
│       ├── class_0_non_agri/                  # Non-agricultural images (3000 files)
│       └── class_1_agri/                      # Agricultural images (3000 files)
├── docs/                                       # Documentation directory
└── venv/                                       # Virtual environment

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
   
📊 Dataset

The project uses a satellite image dataset with two classes:
	•	Class 0: Non-agricultural land (3,000 images)
	•	Class 1: Agricultural land (3,000 images)

Images are stored in the satellite-image-classification/image_dataSAT/ directory.

📚 Learning Objectives

After completing this project, you will be able to:
	1.	Data Management: Efficiently handle large satellite image datasets
	2.	Model Development: Build and train CNN and ViT models from scratch
	3.	Framework Proficiency: Work with both Keras/TensorFlow and PyTorch
	4.	Advanced Architectures: Implement and understand hybrid CNN-ViT models
	5.	Performance Analysis: Evaluate models using multiple metrics and visualizations
	6.	Comparative Analysis: Compare different approaches and frameworks

🔬 Notebooks Overview

1. Data Loading and Preparation
	•	01 Data Loading - Memory vs Generator.ipynb: Compare memory-based vs generator-based data loading
	•	02k Data Preparation - Keras.ipynb: Keras-specific preprocessing and augmentation
	•	02p Data Preparation - PyTorch.ipynb: PyTorch-specific preprocessing and augmentation

2. Model Development
	•	03k Model Development and Evaluation - Keras.ipynb: CNN in Keras/TensorFlow
	•	03p Model Development and Evaluation - PyTorch.ipynb: CNN in PyTorch

3. Comparative Analysis
	•	04 Evaluation - Keras and PyTorch Models.ipynb: Performance comparison

4. Advanced Architectures
	•	05k CNN and Vision Transformers Hybrid - Keras.ipynb
	•	05p CNN and Vision Transformers Hybrid - PyTorch.ipynb
	•	06 Evaluation - CNN and Vision Transformers Hybrids.ipynb

🏗️ Model Architectures

1. Convolutional Neural Networks (CNNs)
	•	Local feature extraction
	•	Efficient for smaller datasets

2. Vision Transformers (ViTs)
	•	Self-attention for global spatial dependencies
	•	Strong performance on complex vision tasks

3. CNN-ViT Hybrid Models
	•	Combine CNN’s local feature extraction with ViT’s global attention
	•	Efficient and high-performing approach

📈 Evaluation Metrics
	•	Accuracy – overall correctness
	•	Precision – correctness of positive predictions
	•	Recall – ability to detect positives
	•	F1-Score – balance between precision & recall
	•	ROC Curves – threshold analysis
	•	Confusion Matrices – detailed classification results

🛠️ Key Features
	•	Side-by-side Keras and PyTorch implementations
	•	Generator-based loading for large datasets
	•	CNN-ViT hybrid architecture implementation
	•	Reproducible experiments with fixed random seeds
	•	Comprehensive evaluation and visualization

📖 Usage
	1.	Start with the data loading notebook
	2.	Follow the sequential numbering for a complete workflow
	3.	Run notebooks in order for proper data flow
	4.	Experiment with hyperparameters and architectures
	5.	Compare results across different approaches

🤝 Contributing

This project is part of an AI Engineer learning path.
Feel free to:
	•	Experiment with new model architectures
	•	Add evaluation metrics
	•	Improve preprocessing techniques
	•	Share insights and results

⸻

📄 License

This project is provided for educational purposes only as part of the AI Engineer learning curriculum.
The code, models, and notebooks are shared to demonstrate skills in deep learning, computer vision, and machine learning engineering.
They are not intended for production use.

Disclaimer

These Jupyter notebooks were created as part of the IBM AI Engineering Professional Certificate on Coursera.

The exercises and instructions are provided by Coursera/IBM.
This repository only contains my personal solutions, code implementations, and notes created while completing the course.

All course materials, instructions, and related intellectual property remain the copyright of Coursera and IBM.

This repository is shared for educational and demonstration purposes only.