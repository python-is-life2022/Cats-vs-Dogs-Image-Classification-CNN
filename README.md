# 🐱🐶 Cats vs Dogs Image Classification with CNN

An end-to-end deep learning pipeline built with **TensorFlow / Keras** for binary image classification on the classic *Kaggle Dogs vs. Cats* dataset.

---

## 📌 Project Overview

This repository focuses on building an optimized Convolutional Neural Network (CNN) pipeline to classify images into **Cat** or **Dog**. 

The current stage covers:
1. **Dataset Organization & Restructuring:** Sorting raw data into class-specific directories (`cat` vs `dog`).
2. **Data Exploration:** Visualizing sample images and inspecting dimensions/channels.
3. **Data Preprocessing & Normalization:** Scaling raw pixel intensities from `[0, 255]` to the `[0.0, 1.0]` range for stable gradient descent and faster convergence.

---

## 🗂️ Project Structure
```text
.
├── cats_vs_dogs_main_file/
│   └── train/
│       ├── cat_train_images/    # 12,500 Cat images
│       └── dog_train_images/    # 12,500 Dog images
├── notebooks/                   # Jupyter / Colab notebooks
│   └── cats_vs_dogs.ipynb
├── requirements.txt             # Environment dependencies
└── README.md                    # Project documentation

⚙️ Tech Stack & Requirements
* Language: Python 3.9+
* Deep Learning Framework: TensorFlow 2.x, Keras
* Computer Vision & Processing: OpenCV, NumPy
* Visualization: Matplotlib
* Installation
* Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn
pip install -r requirements.txt

```
🚀 Data Pipeline & Preprocessing Workflow
1. Directory Setup & Sorting
Raw dataset files are organized into structured directories for compatibility with modern data loaders (e.g., ImageDataGenerator / tf.keras.utils.image_dataset_from_directory):
```
import os
import shutil

train_dir = 'cats_vs_dogs_main_file/train'
labels = ['cat', 'dog']

for label in labels:
os.makedirs(os.path.join(train_dir, f'{label}_train_images'), exist_ok=True)
for i in range(12500):
src = os.path.join(train_dir, f'{label}.{i}.jpg')
dst = os.path.join(train_dir, f'{label}_train_images', f'{label}.{i}.jpg')
if os.path.exists(src):
shutil.move(src, dst)

```
2. Normalization & Preprocessing
To accelerate training and avoid exploding/vanishing gradients, pixel values are normalized:
X 
norm
​
 = 
255.0
X
​
 whereX∈[0,255]

 🛠️ Next Steps & Roadmap
[x] Dataset structuring & directory partitioning
[x] Exploratory Data Analysis (EDA) & inspection
[x] Manual pixel normalization pipeline
[ ] Implement ImageDataGenerator / tf.data pipeline with Data Augmentation (rotation, zoom, flips)
[ ] Build baseline CNN architecture (Conv2D, BatchNorm, MaxPooling2D, Dropout, Dense)
[ ] Train model using Binary Cross-Entropy loss & Adam optimizer
[ ] Evaluate with Confusion Matrix, Accuracy/Loss curves, and ROC-AUC
[ ] Export model weights for inference


👤 Author
Amir Mostafa Kharazi

* Portfolio: simurghprojects.com
* GitHub: @your-github-username
* LinkedIn: Your LinkedIn Profile
