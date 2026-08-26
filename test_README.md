# Cats vs Dogs Image Classification — Model Testing Pipeline

This module tests a trained **Cats vs Dogs CNN model** on test images using TensorFlow and Keras.

---

## 📌 Overview

The testing workflow performs the following steps:
1. **Load Trained Model:** Loads the pretrained `.keras` CNN model.
2. **Probability Wrapper:** Appends a `Softmax` activation layer using a `Sequential` container to convert raw model logits into class probabilities.
3. **Image Preprocessing:** Iterates through the `test_photos/test_photos` directory, loads each image, resizes it to $64 \times 64$, and normalizes pixel values to the range $[0, 1]$.

---

## 🛠️ Code Reference
```python
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model, Sequential
from keras.layers import Softmax

# 1. Load trained model
model = load_model('cat_vs_dog_images_classification_CNN_model (2).keras')

# 2. Wrap model with Softmax for probability outputs
test_model = Sequential([
model,
Softmax()
])

# 3. Preprocess test images
test_images = os.listdir('test_photos/test_photos')
for img in test_images:
image = plt.imread(f'test_photos/test_photos/{img}')
image = tf.image.resize(image, (64, 64))
image = tf.cast(image, tf.float32) / 255.0
