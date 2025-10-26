# AI Engineer Portfolio

[![IBM AI Engineering](https://img.shields.io/badge/IBM-AI%20Engineering-blue)](https://www.coursera.org/professional-certificates/ai-engineer)
[![Python](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)](https://pytorch.org)

This repository contains my AI Engineering portfolio developed as part of the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera.

## 🎯 Portfolio Projects

This repository showcases two comprehensive AI engineering projects:

### 📡 [Satellite Image Classification](./satellite-image-classification/)
**Deep Learning for Computer Vision**

A comprehensive deep learning project for satellite image classification using both Keras/TensorFlow and PyTorch frameworks. The project demonstrates:

- **Convolutional Neural Networks (CNNs)** for local feature extraction
- **Vision Transformers (ViTs)** for global spatial dependencies
- **Hybrid CNN-ViT models** combining both approaches
- Comparative analysis between TensorFlow/Keras and PyTorch
- Complete ML pipeline: data loading, augmentation, training, and evaluation

👉 **[See Project Details](./satellite-image-classification/README.md)**

---

### 🤖 [Document Q&A Bot](./qa-bot/)
**RAG Architecture for Document Intelligence**

A demonstration of Retrieval-Augmented Generation (RAG) architecture for document question-answering systems. The project showcases:

- **Document Processing**: Loading and parsing PDF/text files
- **Text Chunking**: Splitting documents into manageable pieces
- **Vector Storage**: Creating embeddings and storing in Chroma vector database
- **Semantic Search**: Finding relevant document chunks for queries
- **RAG Pipeline**: Combining retrieval with generation using LangChain
- **Web Interface**: Gradio-based user interface

👉 **[See Project Details](./qa-bot/README.md)**

---

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

2. **Navigate to a project directory**
   ```bash
   cd satellite-image-classification  # or cd qa-bot
   ```

3. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**
   - For **Satellite Image Classification**: See [project README](./satellite-image-classification/README.md)
   - For **Q&A Bot**: See [project README](./qa-bot/README.md)

## 📚 Learning Outcomes

These projects demonstrate proficiency in:

- **Deep Learning**: CNNs, Vision Transformers, and hybrid architectures
- **Framework Mastery**: Working with both TensorFlow/Keras and PyTorch
- **Computer Vision**: Image classification, preprocessing, and augmentation
- **NLP & RAG**: Document processing, vector embeddings, semantic search
- **ML Engineering**: End-to-end pipeline development and deployment
- **Model Evaluation**: Comprehensive metrics and comparative analysis
- **Software Engineering**: Clean code, documentation, and best practices

## 🏗️ Repository Structure

```
ai-engineer/
├── satellite-image-classification/    # Deep learning project
│   ├── README.md                      # Project documentation
│   ├── *.ipynb                        # Jupyter notebooks
│   └── [notebooks and models]
│
├── qa-bot/                            # RAG demonstration tool
│   ├── README.md                      # Project documentation
│   ├── qabot.py                       # Main application
│   └── [other files]
│
└── README.md                          # This file
```

## 🛠️ Technologies & Tools

- **Deep Learning**: TensorFlow, Keras, PyTorch
- **Computer Vision**: CNNs, Vision Transformers, Image Processing
- **NLP**: LangChain, Transformers, HuggingFace
- **Vector Databases**: Chroma
- **Web Frameworks**: Gradio
- **Data Science**: Pandas, NumPy, Matplotlib, Seaborn
- **ML Tools**: Scikit-learn, Jupyter

## 📄 License & Attribution

This portfolio is provided for **educational purposes** as part of the AI Engineer learning curriculum.

### ⚠️ Important Notice

- **Course materials and instructions** are provided by Coursera/IBM
- **This repository** contains only my personal solutions, code implementations, and notes
- **All course materials** and related intellectual property remain the copyright of Coursera and IBM
- **Not intended for production use** - Educational demonstration only

## 🙏 Acknowledgements

- Projects developed as part of the [IBM AI Engineering Professional Certificate](https://www.coursera.org/professional-certificates/ai-engineer) on Coursera
- Special thanks to IBM instructors and the Coursera team for providing excellent course materials
- Course exercises and instructions provided by Coursera/IBM
- This repository contains my personal solutions, code implementations, and notes created while completing the course

---

**Built with ❤️ as part of the AI Engineer Professional Certificate**
