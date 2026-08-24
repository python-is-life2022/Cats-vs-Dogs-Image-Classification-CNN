# Cats vs Dogs Image Classification with CNN

An end-to-end deep learning project for binary image classification using a Convolutional Neural Network (CNN). The model is designed to classify images as either a cat or a dog using TensorFlow and Keras.

## Project Overview

This project covers the initial stages of a Cats vs Dogs image-classification pipeline:

- Inspecting image dimensions and sample images
- Organizing the raw dataset into class-specific directories
- Separating cat and dog images
- Converting image pixel values from the `[0, 255]` range to `[0.0, 1.0]`
- Preparing the dataset for CNN training

Image normalization is performed using the following formula:

```text
normalized_pixel = pixel / 255.0
```

This keeps input values in a smaller and consistent range, which generally helps neural-network training become more stable.

## Dataset Structure

After organizing the files, the expected directory structure is:

```text
cats-vs-dogs-cnn/
├── cats_vs_dogs_main_file/
│   └── train/
│       ├── cat_train_images/
│       │   ├── cat.0.jpg
│      n│       │   ├── cat.0.jpg
│       │  
│       └── dog_train_images/
│           ├── dog.0.jpg
│           ├── dog.1.jpg
│           └── ...
├── notebooks/
├── requirements.txt
└── README.md
```

The
```

The,500 cat images and 12,500 dog images.

## Technologies

- Python 3.9+
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Pandas
- Jupyter Notebook or Google Colab

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/cats-vs-dogs-cnn.git
cd cats-vs-dogs-cnn

```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```text
tensorflow>=2.10.0
keras
opencv-python
numpy
matplotlib
pandas
```

## Data Organization

The raw images are moved into separate directories according to their labels:

```python
import os
import shutil

train_dir = os.path.join("cats_vs_dogs_main_file", "train")

for label in ["cat", "dog"]:
    target_dir = os.path.join(train_dir, f"{label}_train_images")
    os.makedirs(target_dir, exist_ok=True)

    for index in range(12500):
        source = os.path.join(train_dir, f"{label}.{index}.jpg")
        destination = os.path.join(target_dir, f"{label}.{index}.jpg")

        if os.path.exists(source):
            shutil.move(source, destination)
```

`exist_ok=True` prevents an error if the target directory already exists. The `os.path.exists` check prevents the script from failing when a file is missing or has already been moved.

## Image Inspection

A sample image can be loaded and displayed with Matplotlib:

```python
import os
import matplotlib.pyplot as plt

image_path = os.path.join(
    "cats_vs_dogs_main_file",
    "train",
    "cat_train_images",
    "cat.1.jpg",
)

image = plt.imread(image_path)

plt.axis("off")
plt.imshow(image)
plt.show()

print("Image shape:", image.shape)
```

## Image Normalization

Pixel normalization can be applied as follows:

```python
import os
import matplotlib.pyplot as plt

train_dir = os.path.join("cats_vs_dogs_main_file", "train")

for folder in ["cat_train_images", "dog_train_images"]:
    folder_path = os.path.join(train_dir, folder)

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = plt.imread(image_path).astype("float32")
        normalized_image = image / 255.0
```

The variable `normalized_image` contains floating-point pixel values in the `[0.0, 1.0]` range.

This loop demonstrates the normalization process, but it does not save the normalized images. For model training, it is usually more efficient to normalize images while loading them instead of creating duplicate image files in memory or on disk.

## Recommended Training Pipeline

For a scalable training workflow, use `ImageDataGenerator` or a `tf.data` pipeline:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_size = (180, 180)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
)

train_generator = train_datagen.flow_from_directory(
    "cats_vs_dogs_main_file/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="training",
)

validation_generator = train_datagen.flow_from_directory(
    "cats_vs_dogs_main_file/train",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
    subset="validation",
)
```

With `rescale=1.0 / 255.0`, normalization is applied automatically when each image is loaded.

## Planned CNN Architecture

The project imports the main building blocks needed for a CNN model:

- `Conv2D` for extracting visual features
- `BatchNormalization` for more stable training
- `Activation` for nonlinear transformations
- `MaxPooling2D` for reducing spatial dimensions
- `GlobalAveragePooling2D` for reducing feature maps
- `Dropout` for reducing overfitting
- ` loss function for classification
- `Adam` as the optimizer
- Binary cross-entropy as the loss function

A future baseline model can be built with the following structure:

```text
Input image
    ↓

Input image
    ↓

    ↓
Max Pooling
    ↓
Convolution + Batch Normalization + Activation
    ↓
Max Pooling
    ↓
Global Average Pooling
    ↓
Dropout
    ↓
Dense output layer with sigmoid activation
```

## Roadmap

- [x] Inspect sample images and image dimensions
- [x] Separate cat and dog images into directories
- [x] Normalize pixel values to the `[0.0, 1.0]` range
- [ ] Build the baseline CNN model
- [ ] Add Plot accuracy and loss curves
- [ Train the model with binary cross-entropy
- [ ] Plot accuracy and loss curves
- [ ] Evaluate the model with a confusion matrix
- [ ] Calculate precision, recall, and ROC-AUC
- [ ] Save the trained model
- [ ] Add an inference script for new images
- [ ] Add data augmentation and model tuning

## Important Notes

1. Run the file organization script only once, or make it idempotent as shown above.
2. The folder passed to `flow_from_directory` must contain one subdirectory per class.
3. The manual normalization loop creates a subdirectory per class.
3. The manual normalization loop creates a, normalize during loading with `rescale=1.0 / 255.0` or a Keras `Rescaling` layer.
5. Keep training, validation, and test data separated to avoid data leakage.

## Author

**Amir Mostafa Kharazi**

- Portfolio: [simurghprojects.com](https://simurghprojects.com)
- GitHub: [your-github-username](https://github.com/your-username)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)

## License

This project is available under the MIT License. Add a `LICENSE` file to the repository if you want to distribute the project under that license.

همین 😐😐😐😐😐😐
